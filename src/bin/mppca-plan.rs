use std::{fs::File, path::PathBuf};
use clap::Parser;
use kmppca::Plan;

#[derive(clap::Parser)]
struct Args {
    plan_file_out:PathBuf,
    n_volumes:usize,
    n_processes:usize,
    #[clap(value_delimiter = ',',short = 'd')]
    volume_dims:Vec<usize>,
    #[clap(value_delimiter = ',',short = 'p')]
    patch_size:Vec<usize>,
    #[clap(value_delimiter = ',',short = 's')]
    patch_stride:Vec<usize>,
}



fn main() {

    let args = Args::parse();

    {   // test if we can write the file before spending time building the patch planner
        let _ = File::create(&args.plan_file_out).expect("failed to create file");
    }
    
    let mut volume_dims = [0;3];
    volume_dims.as_mut_slice().copy_from_slice(&args.volume_dims[0..3]);

    let mut patch_size = [0;3];
    patch_size.as_mut_slice().copy_from_slice(&args.patch_size[0..3]);

    let mut patch_stride = [0;3];
    patch_stride.as_mut_slice().copy_from_slice(&args.patch_stride[0..3]);

    Plan::new(
        volume_dims,
        args.n_volumes,
        patch_size,
        patch_stride,
        args.n_processes
    ).to_file(&args.plan_file_out);

}