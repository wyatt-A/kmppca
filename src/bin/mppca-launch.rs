use std::path::PathBuf;

use cfl::{ndarray::{parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator}, Array3, Axis, ShapeBuilder}, CflReader, CflWriter};
use cfl::num_complex::Complex32;
use clap::Parser;
use kmppca::{ceiling_div, singular_value_threshold_mppca, Config, Plan};

#[derive(clap::Parser)]
struct Args {
    config_file:PathBuf,
    process_index:usize,
    partition_index:usize,
}

fn main() {

    let args = Args::parse();

    let config = Config::from_file(args.config_file);
    let plan = Plan::from_file(&config.patch_plan_file);

    let patches_to_process = &plan.plan_data[args.partition_index][args.process_index];
    let n_patches_to_process = patches_to_process.len();

    let n_batches = ceiling_div(n_patches_to_process, config.batch_size);

    println!("n_batches = {}",n_batches);
    println!("n total patches = {}",n_patches_to_process);
    
    let mut writers:Vec<_> = config.output_files.iter().map(|path| CflWriter::open(path).unwrap() ).collect();
    let readers:Vec<_> = config.input_files.iter().map(|path| CflReader::new(path).unwrap() ).collect();

    let write_operation = |a,b| a + b;

    for (batch_id,batch) in patches_to_process.chunks(config.batch_size).enumerate() {

        println!("loading patch data for batch {} of {} ...",batch_id+1,n_batches);
        let mut patch_data = Array3::from_elem((plan.patch_generator.patch_size(),plan.n_volumes,batch.len()).f(), Complex32::ZERO);
        patch_data.axis_iter_mut(Axis(2)).enumerate().for_each(|(idx,mut patch)|{
            // get the patch index values, shared over all volumes in data set
            let patch_indices = plan.patch_generator.nth(batch[idx]).unwrap();
            // iterate over each volume, assigning patch data to a single column
            for (mut col,vol) in patch.axis_iter_mut(Axis(1)).zip(readers.iter()) {
                // read_into uses memory mapping internally to help with random indexing of volume files
                vol.read_into(&patch_indices, col.as_slice_memory_order_mut().unwrap()).unwrap()
            }
        });

        println!("doing MPPCA denoising on batch {} of {} ...",batch_id+1,n_batches);
        singular_value_threshold_mppca(&mut patch_data, None);

        println!("writing data for batch {} of {} ...", batch_id+1, n_batches);
        patch_data.axis_iter_mut(Axis(2)).enumerate().for_each(|(idx,mut patch)|{
            // get the patch index values, shared over all volumes in data set
            let patch_indices = plan.patch_generator.nth(batch[idx]).unwrap();
            // iterate over each volume, assigning patch data to a single column
            for (mut col,vol) in patch.axis_iter_mut(Axis(1)).zip(writers.iter_mut()) {
                // write_from uses memory mapping internally to help with non-contiguous indexing of volume files
                vol.write_op_from(&patch_indices, col.as_slice_memory_order_mut().unwrap(),write_operation).unwrap()
            }
        });

    }
}