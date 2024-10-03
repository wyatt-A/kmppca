use std::path::PathBuf;
use cfl::{ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator}, CflReader, CflWriter};
use clap::Parser;
use kmppca::{Config, Plan};


#[derive(clap::Parser)]
struct Args {
    input_plan:PathBuf,
    output_config:PathBuf,
    input_file_pattern:String,
    batch_size:usize,
}

fn main() {

    // let config = PathBuf::from("config");
    // let plan_file = PathBuf::from("plan");
    // let file_pattern = "/privateShares/wa41/24.chdi.01.test/240725-3-1/downsampled/i*.cfl";
    // let batch_size = 1000;

    let args = Args::parse();

    let plan = Plan::from_file(&args.input_plan);


    println!("checking input files ...");
    let input_files:Vec<_> = (0..plan.n_volumes).into_par_iter().map(|i|{
        let filename = PathBuf::from(args.input_file_pattern.replace("*", &format!("{:01}",i)));
        //println!("reading {}",filename.display());
        CflReader::new(&filename).expect("failed to read file");
        filename
    }).collect();

    println!("pre-allocating output files ...");
    let output_files:Vec<_> = (0..plan.n_volumes).into_par_iter().map(|i|{
        let filename = PathBuf::from(args.input_file_pattern.replace("*", &format!("d{:01}",i)));
        //println!("pre-allocating {:?}",filename);
        CflWriter::new(&filename,&plan.volume_dims).unwrap();
        filename
    }).collect();
    println!("pre-allocated {} files.",plan.n_volumes);

    Config {
        patch_plan_file: args.input_plan.to_owned(),
        input_files,
        output_files,
        batch_size: args.batch_size,
    }
    .to_file(args.output_config);

    println!("done.");

}
