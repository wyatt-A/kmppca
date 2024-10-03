use std::path::PathBuf;
use cfl::{ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator}, CflReader, CflWriter};
use clap::Parser;
use kmppca::{Config, Plan};

#[derive(clap::Parser)]
struct Args {
    input_plan:PathBuf,
}


fn main() {
    let args = Args::parse();
    let plan = Plan::from_file(args.input_plan);
    println!("n_processes: {}",plan.n_processes);
    println!("n_partitions: {}",plan.plan_data.len());
} 