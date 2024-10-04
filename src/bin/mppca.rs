use std::{path::{Path, PathBuf}, process::Command};
use cfl::{ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator}, CflReader, CflWriter};
use clap::Parser;
use kmppca::{slurm::SlurmTask, Config, Plan};

#[derive(clap::Parser)]
struct Args {
    input_plan:PathBuf,
    output_config:PathBuf,
    work_dir:PathBuf,
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
    .to_file(&args.output_config);

    launch_array_jobs(&args.work_dir, "denoise_test", args.output_config, &plan);
    
    println!("job launch complete.");

}


fn launch_array_jobs(work_dir:impl AsRef<Path>,job_name:&str,config:impl AsRef<Path>,plan:&Plan) {

    let n_partitions = plan.plan_data.len();
    let mut n_processes = plan.plan_data[0].len();

    let mut cmd = Command::new("mppca-launch");
    cmd.arg(config.as_ref());
    cmd.arg("$SLURM_ARRAY_TASK_ID");
    cmd.arg(0.to_string());

    let mut task = SlurmTask::new(work_dir.as_ref(),job_name,1_000)
    .output(work_dir.as_ref())
    .array(0, n_processes-1)
    .command(cmd);

    let mut jid = task.submit();

    for p in 1..n_partitions {
        n_processes = plan.plan_data[p].len();

        let mut cmd = Command::new("mppca-launch");
        cmd.arg(config.as_ref());
        cmd.arg("$SLURM_ARRAY_TASK_ID");
        cmd.arg(p.to_string());
    
        let mut task = SlurmTask::new(work_dir.as_ref(),job_name,1_000)
        .output(work_dir.as_ref())
        .array(0, n_processes-1)
        .job_dependency_after_ok(jid)
        .command(cmd);

        jid = task.submit();

    }





    
    


}