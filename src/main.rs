//! RockDuck CLI binary

use clap::{Parser, Subcommand};
use rockduck::{Result, RockDuck};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "rockduck",
    about = "RockDuck - HTAP Embedded Database",
    version = "0.2.0"
)]
struct Cli {
    /// 数据目录
    #[arg(short, long, default_value = "./data")]
    data_dir: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 打开数据库并输出信息
    Info,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Command::Info => {
            let _db = RockDuck::open(&cli.data_dir)?;
            println!("RockDuck opened successfully at {:?}", cli.data_dir);
            match _db.next_txn_id() {
                Ok(id) => println!("Transaction counter: {}", id),
                Err(e) => eprintln!("Transaction counter error: {}", e),
            }
        }
    }

    Ok(())
}
