//! RockDuck - 嵌入式列存数据库
//!
//! 基于 RocksDB + Vortex + DuckDB 的列式存储引擎

use clap::{Parser, Subcommand, ValueEnum};
use rockduck::RockDuck;
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(
    name = "rockduck",
    about = "RockDuck - 嵌入式列存数据库",
    version = "0.1.0",
    author = "RockDuck Team"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// 数据库路径
    #[arg(short, long, default_value = "./rockduck_data")]
    data_dir: PathBuf,

    /// 日志级别
    #[arg(short, long, value_enum, default_value_t = LogLevel::Info)]
    verbose: LogLevel,
}

#[derive(Clone, ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}

impl From<LogLevel> for Level {
    fn from(l: LogLevel) -> Self {
        match l {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// 插入记录
    Insert {
        /// 主键
        pk: String,
        /// 列值，格式: col1=val1,col2=val2
        #[arg(value_parser = parse_kv_pairs, default_value = "")]
        columns: Vec<ColumnValue>,
        /// 表名
        #[arg(short, long, default_value = "default")]
        table: String,
    },
    /// 根据主键获取记录
    Get {
        /// 主键
        pk: String,
        /// 表名
        #[arg(short, long, default_value = "default")]
        table: String,
    },
    /// 扫描记录范围
    Scan {
        /// 起始主键（可选）
        #[arg(short, long)]
        start: Option<String>,
        /// 结束主键（可选）
        #[arg(short, long)]
        end: Option<String>,
        /// 表名
        #[arg(short, long, default_value = "default")]
        table: String,
    },
    /// 删除记录
    Delete {
        /// 主键
        pk: String,
        /// 表名
        #[arg(short, long, default_value = "default")]
        table: String,
    },
    /// 显示表统计信息
    Stats {
        /// 表名
        #[arg(short, long, default_value = "default")]
        table: String,
    },
    /// 数据库信息
    Info,
}

#[derive(Clone, Debug)]
struct ColumnValue {
    name: String,
    value: i64,
}

fn parse_kv_pairs(s: &str) -> Result<ColumnValue, String> {
    if s.is_empty() {
        return Ok(ColumnValue {
            name: String::new(),
            value: 0,
        });
    }
    if let Some((name, val)) = s.split_once('=') {
        let value = val
            .parse()
            .map_err(|_| format!("Invalid integer: {}", val))?;
        Ok(ColumnValue {
            name: name.to_string(),
            value,
        })
    } else {
        Err("Expected format: name=value".to_string())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // 初始化日志
    let level: Level = cli.verbose.into();
    FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .init();

    info!("Starting RockDuck (data_dir: {:?})", cli.data_dir);

    // 创建或打开数据库
    let db = RockDuck::open(&cli.data_dir)?;
    info!("RockDuck opened successfully");

    match &cli.command {
        Command::Insert { pk, columns, table } => {
            let mut cols = std::collections::HashMap::new();
            for col in columns {
                let arr = arrow_array::Int64Array::from(vec![col.value]);
                cols.insert(
                    col.name.clone(),
                    std::sync::Arc::new(arr) as std::sync::Arc<dyn arrow_array::Array>,
                );
            }
            match db.insert(table, pk.as_bytes(), &cols) {
                Ok(txn_id) => println!("Inserted with txn_id: {}", txn_id),
                Err(e) => {
                    eprintln!("Insert error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
        Command::Get { pk, table } => match db.get(table, pk.as_bytes()) {
            Ok(Some(record)) => {
                println!(
                    "Found record: {} rows, {} columns",
                    record.num_rows(),
                    record.num_columns()
                );
                println!("{:?}", record);
            }
            Ok(None) => println!("Record not found"),
            Err(e) => {
                eprintln!("Get error: {:?}", e);
                std::process::exit(1);
            }
        },
        Command::Scan { start, end, table } => {
            let pk_range = match (start, end) {
                (Some(s), Some(e)) => Some((s.as_bytes().to_vec(), e.as_bytes().to_vec())),
                (Some(s), None) => Some((s.as_bytes().to_vec(), vec![])),
                (None, Some(e)) => Some((vec![], e.as_bytes().to_vec())),
                (None, None) => None,
            };
            match db.scan(table, pk_range, None) {
                Ok(batches) => {
                    println!("Scanned {} record batch(es)", batches.len());
                    for batch in batches {
                        println!(
                            "  Batch: {} rows, {} columns",
                            batch.num_rows(),
                            batch.num_columns()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Scan error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
        Command::Delete { pk, table } => match db.delete(table, pk.as_bytes()) {
            Ok(_) => println!("Deleted record: {}", pk),
            Err(e) => {
                eprintln!("Delete error: {:?}", e);
                std::process::exit(1);
            }
        },
        Command::Stats { table } => match db.get_table_stats(table) {
            Ok(Some(stats)) => {
                println!("Table: {}", stats.table);
                println!("  Row count: {}", stats.row_count);
                println!("  Deleted rows: {}", stats.deleted_rows);
                println!("  Alive rows: {}", stats.alive_rows());
                println!("  Segment count: {}", stats.segment_count);
                println!("  Total size: {} bytes", stats.total_size);
                println!("  Compressed size: {} bytes", stats.compressed_size);
                println!("  Deletion ratio: {:.2}%", stats.del_ratio() * 100.0);
            }
            Ok(None) => println!("No stats found for table: {}", table),
            Err(e) => {
                eprintln!("Stats error: {:?}", e);
                std::process::exit(1);
            }
        },
        Command::Info => {
            let info = db.get_info();
            println!("RockDuck Information:");
            println!("  Data directory: {:?}", info.data_dir);
            println!("  Transaction counter: {}", info.txn_counter);
        }
    }

    info!("Shutting down RockDuck");
    Ok(())
}
