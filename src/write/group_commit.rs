//! CAS-based Leader Election for Group Commit.
//!
//! Runs atop `durability`'s `walog::WalWriter`, which already provides mutex-protected
//! writes. This module adds cross-thread coordination via CAS operations so that
//! only one thread performs the flush while others wait on the atomic flag.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// CAS-based leader election for group commit.
///
/// Uses `Mutex<Option<Instant>>` for the last sync timestamp instead of `UnsafeCell`
/// to provide safe interior mutability without requiring `unsafe impl Sync`.
pub struct CasLeaderElection {
    leader_active: AtomicBool,
    waiter_count: AtomicU64,
    last_sync_time: parking_lot::Mutex<Option<Instant>>,
}

impl CasLeaderElection {
    pub fn new() -> Self {
        Self {
            leader_active: AtomicBool::new(false),
            waiter_count: AtomicU64::new(0),
            last_sync_time: parking_lot::Mutex::new(Some(Instant::now())),
        }
    }

    pub fn try_claim_leader(&self) -> bool {
        self.leader_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn release_leadership(&self) {
        // Use lock() without unwrap - parking_lot mutexes don't poison on panic
        // but the lock acquisition itself can fail in rare cases
        if let Some(mut guard) = self.last_sync_time.try_lock() {
            *guard = Some(Instant::now());
        }
        self.leader_active.store(false, Ordering::Release);
    }

    pub fn wait_for_sync(&self) {
        while self.leader_active.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }

    pub fn add_waiter(&self) {
        self.waiter_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn remove_waiter(&self) {
        self.waiter_count.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn waiter_count(&self) -> u64 {
        self.waiter_count.load(Ordering::Relaxed)
    }

    pub fn is_leader_active(&self) -> bool {
        self.leader_active.load(Ordering::Acquire)
    }

    pub fn ms_since_last_sync(&self) -> u64 {
        let guard = self.last_sync_time.lock();
        guard.map_or(0, |i| i.elapsed().as_millis() as u64)
    }
}

impl Default for CasLeaderElection {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct GroupCommitSettings {
    pub min_waiters: u64,
    pub max_wait_ms: u64,
}

impl Default for GroupCommitSettings {
    fn default() -> Self {
        Self {
            min_waiters: 2,
            max_wait_ms: 50,
        }
    }
}

pub fn should_trigger_group_commit(
    election: &CasLeaderElection,
    settings: &GroupCommitSettings,
) -> bool {
    election.waiter_count() >= settings.min_waiters
        || election.ms_since_last_sync() >= settings.max_wait_ms
}

pub fn spawn_leader_thread<F>(
    election: Arc<CasLeaderElection>,
    settings: GroupCommitSettings,
    flush_fn: F,
) -> std::thread::JoinHandle<()>
where
    F: Fn() + Send + 'static,
{
    std::thread::spawn(move || loop {
        while !should_trigger_group_commit(&election, &settings) {
            std::thread::sleep(Duration::from_millis(1));
        }

        if election.try_claim_leader() {
            (flush_fn)();
            election.release_leadership();

            if election.waiter_count() == 0 {
                break;
            }
        } else {
            election.wait_for_sync();

            if election.waiter_count() == 0 {
                break;
            }
        }
    })
}

// ---------------------------------------------------------------------------
