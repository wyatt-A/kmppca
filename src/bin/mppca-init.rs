use std::path::{Path, PathBuf};
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

    // used to store the noise component removed from the data set
    let noise_volume = Path::new(&args.input_file_pattern).with_file_name("noise");
    CflWriter::new(&noise_volume,&plan.volume_dims).unwrap();

    // used to track the number off additions to compute the patch average
    let normalization_volume = Path::new(&args.input_file_pattern).with_file_name("norm");
    CflWriter::new(&normalization_volume,&plan.volume_dims).unwrap();

    Config {
        patch_plan_file: args.input_plan.to_owned(),
        input_files,
        output_files,
        batch_size: args.batch_size,
        noise_volume,
        normalization_volume,
    }
    .to_file(args.output_config);

    println!("done.");

}
