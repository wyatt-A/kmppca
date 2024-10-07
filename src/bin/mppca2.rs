use std::{fmt::write, fs::File, io::{Read, Write}, os::unix::process, path::{Path, PathBuf}, process::Command, sync::{Arc, Mutex}, time::Instant};

use bincode::de::read;
use cfl::{ndarray::{parallel::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator}, s, Array1, Array3, ArrayD, Axis, ShapeBuilder, Slice}, num_complex::ComplexFloat, CflReader, CflWriter};
use fourier::window_functions::WindowFunction;
use kmppca::{ceiling_div, patch_planner::PatchPlanner, singular_value_threshold_mppca, slurm::{self, SlurmTask}};
use ndarray_linalg;
use cfl::num_complex::Complex32;
use serde::{Deserialize, Serialize};

use clap::{Parser,Subcommand};

#[derive(Parser, Debug)]
struct Args {
    #[command(subcommand)]
    action: SubCmd
}

#[derive(Subcommand, Debug)]
enum SubCmd {
    Plan(PlanArgs),
    Init(InitArgs),
    Denoise(DenoiseArgs),
    DenoisePartition(DenoisePartitionArgs)
}

#[derive(Parser, Debug, Clone)]
struct PlanArgs {
    work_dir:PathBuf,
    input_cfl_pattern:String,
    n_volumes:usize,
    n_processes:usize,
    batch_size:usize,
    #[clap(short = 'p')]
    patch_size:Vec<usize>,
    #[clap(short = 's')]
    patch_stride:Vec<usize>,
}

#[derive(Parser, Debug, Clone)]
struct InitArgs {
    work_dir:PathBuf,
}

#[derive(Parser, Debug, Clone)]
struct DenoiseArgs {
    work_dir:PathBuf,
}

#[derive(Parser, Debug, Clone)]
struct DenoisePartitionArgs {
    work_dir:PathBuf,
    partition_id:usize,
    process_id:usize,
}


fn main() {

    let args = Args::parse();

    match args.action {
        SubCmd::Plan(plan_args) => {

            let mut patch_size = [0usize;3];
            let mut patch_stride = [0usize;3];
            patch_size.iter_mut().zip(plan_args.patch_size).for_each(|(a,b)| *a = b);
            patch_stride.iter_mut().zip(plan_args.patch_stride).for_each(|(a,b)| *a = b);

            mppca_plan(
                plan_args.work_dir,
                &plan_args.input_cfl_pattern,
                plan_args.n_volumes,
                patch_size,
                patch_stride,
                plan_args.n_processes,
                plan_args.batch_size
            );
        },
        SubCmd::Init(init_args) => {
            mppca_build_4d(&init_args.work_dir);
            mppca_phase_correct(&init_args.work_dir);
            mppca_pad(&init_args.work_dir);
        },
        SubCmd::Denoise(denoise_args) => {

            let c = Config::from_file(denoise_args.work_dir.join("config"));

            let denoise_out = denoise_args.work_dir.join("denoised");
            let noise_out = denoise_args.work_dir.join("noise");
            let norm_out = denoise_args.work_dir.join("norm");

            let _ = CflWriter::new(denoise_out,&c.padded_stack_size).unwrap();
    
            let _ = CflWriter::new(noise_out,&c.patch_planner.padded_array_size()).unwrap();
            let _ = CflWriter::new(norm_out,&c.patch_planner.padded_array_size()).unwrap();

            if slurm::is_installed() {
                launch_array_jobs(&denoise_args.work_dir,"denoise_test")
            }else {
                let n_partitions = c.patch_planner.n_total_partitions();
                println!("n partitions: {}",n_partitions);
                println!("n processes: {}",c.n_processes);
                for partition in 0..n_partitions {
                    for process in 0..c.n_processes {
                        mppca_denoise(&denoise_args.work_dir, process, partition);
                    }
                }
            }
        },
        SubCmd::DenoisePartition(denoise_partition_args) => {
            mppca_denoise(
                denoise_partition_args.work_dir,
                denoise_partition_args.process_id,
                denoise_partition_args.partition_id
            );
        },
    }
}

fn launch_array_jobs(work_dir:impl AsRef<Path>,job_name:&str) {

    let c = Config::from_file(work_dir.as_ref().join("config"));

    let n_partitions = c.patch_planner.n_total_partitions();

    let patch_ids = c.patch_planner.partition_patch_ids(c.n_processes, 0);

    let mut n_processes = patch_ids.len();

    let this_exe = std::env::current_exe().expect("failed to get this exe");

    let mut cmd = Command::new(&this_exe);
    cmd.arg("denoise-partition");
    cmd.arg(work_dir.as_ref());
    cmd.arg("$SLURM_ARRAY_TASK_ID");
    cmd.arg(0.to_string());

    let mut task = SlurmTask::new(work_dir.as_ref(),job_name,1_000)
    .output(work_dir.as_ref())
    .array(0, n_processes-1)
    .command(cmd);

    let mut jid = task.submit();

    for partition in 1..n_partitions {

        let patch_ids = c.patch_planner.partition_patch_ids(c.n_processes, partition);
        n_processes = patch_ids.len();

        let mut cmd = Command::new(&this_exe);
        cmd.arg("denoise-partition");
        cmd.arg(work_dir.as_ref());
        cmd.arg("$SLURM_ARRAY_TASK_ID");
        cmd.arg(0.to_string());
    
        let mut task = SlurmTask::new(work_dir.as_ref(),job_name,1_000)
        .output(work_dir.as_ref())
        .array(0, n_processes-1)
        .job_dependency_after_ok(jid)
        .command(cmd);

        jid = task.submit();

    }

}

// fn main() {

//     // 4-D data set of complex images over q-space
//     // do phase-corrections

//     let inputs = "/home/wyatt/test_data/raw";
//     let n_volumes = 67;

//     let dims = [197,120,120,n_volumes];

//     println!("allocating 4-D stack...");
//     let mut writer = cfl::CflWriter::new("test4d",&dims).unwrap();

//     let volume_stride:usize = dims[0..3].iter().product();
    
//     for i in 0..n_volumes {
//         let cfl_in = Path::new(inputs).join(format!("i{:02}",i));
//         let volume_data = cfl::to_array(cfl_in, true).expect("failed to read cfl volume");
//         let volume_data_slice = volume_data.as_slice_memory_order()
//         .expect("cfl volume is not in memory order");
//         let starting_addr = i*volume_stride;
//         println!("writing volume {} of {}",i+1,n_volumes);
//         writer.write_slice(starting_addr,volume_data_slice).expect("failed to write volume into 4-D stack");
//     }
    


//     // do phase corrections

//     let reader = cfl::CflReader::new("test4d").expect("failed to create 4-D cfl reader");

//     let mut tmp_vol = ArrayD::<Complex32>::zeros(dims[0..3].f());
//     let mut pc_tmp = ArrayD::<Complex32>::zeros(dims[0..3].f());

//     // define window function for phase corrections
//     let w = fourier::window_functions::HanningWindow::new(&dims[0..3]);

//     for i in 0..n_volumes {

//         println!("filtering volume {} of {}",i+1,n_volumes);
//         let starting_addr = i*volume_stride;

//         // read volume data
//         let volume_data = tmp_vol.as_slice_memory_order_mut().expect("failed to get memory order slice");
//         reader.read_slice(starting_addr, volume_data).expect("failed to read from 4-D");

//         // phase correct volume
//         pc_tmp.assign(&tmp_vol);
//         fourier::fftw::fftn_mut(&mut tmp_vol);
//         w.apply(&mut tmp_vol);
//         fourier::fftw::ifftn_mut(&mut tmp_vol);
//         tmp_vol.par_mapv_inplace(|x| (x / x.abs()).conj() );
//         pc_tmp *= &tmp_vol;

//         // write volume to 4-D
//         let volume_data = pc_tmp.as_slice_memory_order_mut().expect("failed to get memory order slice");
//         writer.write_slice(starting_addr, volume_data).expect("failed to write to 4-D stack");

//     }

//     let vol_size = [dims[0],dims[1],dims[2]];
//     let patch_planner = PatchPlanner::new(vol_size,[10,10,10],[10,10,10]);
//     let padded_size = patch_planner.padded_array_size();
//     let pad_amount = [padded_size[0] - vol_size[0],padded_size[1] - vol_size[1],padded_size[2] - vol_size[2]];

//     println!("pad amount: {:?}",pad_amount);
//     // do padding

//     let writer = Arc::new(Mutex::new(
//         CflWriter::new("padded",&[padded_size[0],padded_size[1],padded_size[2],n_volumes]).unwrap()
//     ));

//     let reader = CflReader::new("test4d").unwrap();

//     let padded_volume_stride = padded_size.iter().product::<usize>();

//     let mut non_padded_tmp = Array3::<Complex32>::zeros(vol_size.f());
//     let mut padded_tmp = Array3::<Complex32>::zeros(padded_size.f());

//     for vol in 0..n_volumes {

//         println!("padding vol {} of {}",vol+1,n_volumes);
//         // calculate offset addresses
//         let starting_addr_non_padded = vol*volume_stride;
//         let starting_addr_padded = vol*padded_volume_stride;

//         reader.read_slice(starting_addr_non_padded, non_padded_tmp.as_slice_memory_order_mut().unwrap()).unwrap();

//         // assign the non-padded volume to padded volume
//         padded_tmp.slice_mut(s![0..vol_size[0],0..vol_size[1],0..vol_size[2]])
//         .assign(&non_padded_tmp);

//         for ax in 0..3 {
//             if pad_amount[ax] > 0 {
//                 padded_tmp.slice_axis_mut(Axis(ax), Slice::from(vol_size[ax]..))
//                 .assign(
//                     &non_padded_tmp.slice_axis(Axis(ax), Slice::from(0..pad_amount[ax]))
//                 );
//             }
//         }

//         writer.lock().unwrap().write_slice(starting_addr_padded, padded_tmp.as_slice_memory_order().unwrap()).unwrap();
//     }


//     // do mppca denoising

//     let reader = CflReader::new("padded").unwrap();
//     let mut writer = CflWriter::new("output",&[padded_size[0],padded_size[1],padded_size[2],n_volumes]).unwrap();
    
//     let mut noise_writer = CflWriter::new("noise",&[padded_size[0],padded_size[1],padded_size[2]]).unwrap();
//     let mut norm = CflWriter::new("norm",&[padded_size[0],padded_size[1],padded_size[2]]).unwrap();

//     let batch_size = 1000;

//     let mut patch_indixes = vec![0usize;patch_planner.patch_size()];


//     let accum = |a,b| a + b;
//     //let noise_write_op = |_,b| b;


//     let n_partitions = patch_planner.n_total_partitions();
//     for partition in 0..n_partitions {
//         println!("working on partition {} of {}..",partition+1,n_partitions);

//         let n_patches = patch_planner.n_total_partition_patches_lin(partition);

//         let patch_ids:Vec<_> = (0..n_patches).collect();

//         let n_batches = ceiling_div(n_patches, batch_size);

//         for (batch_id,batch) in patch_ids.chunks(batch_size).enumerate() {

//             let mut patch_data = Array3::<Complex32>::zeros((patch_planner.patch_size(),n_volumes,batch.len()).f());
//             let mut noise_data = Array1::from_elem(batch.len().f(),0f32);
//             let mut patch_noise_tmp = Array1::<Complex32>::zeros(patch_planner.patch_size().f());
//             let ones = Array1::<Complex32>::ones(patch_planner.patch_size().f());

//             patch_data.axis_iter_mut(Axis(2)).zip(batch).for_each(|(mut patch,&patch_idx)|{
//                 // get the patch index values, shared over all volumes in data set
//                 patch_planner.linear_indices(partition, patch_idx, &mut patch_indixes);

//                 for mut col in patch.axis_iter_mut(Axis(1)) {                    
//                     reader.read_into(&patch_indixes, col.as_slice_memory_order_mut().unwrap()).unwrap();
//                     patch_indixes.par_iter_mut().for_each(|idx| *idx += padded_volume_stride);
//                 }

//             });

//             println!("doing MPPCA denoising on batch {} of {} ...",batch_id+1,n_batches);
//             singular_value_threshold_mppca(&mut patch_data, noise_data.as_slice_memory_order_mut().unwrap(), None);
            
//             patch_data.axis_iter(Axis(2)).zip(batch).zip(noise_data.iter()).for_each(|((patch,&patch_idx),&patch_noise)|{
//                 // get the patch index values, shared over all volumes in data set
//                 patch_planner.linear_indices(partition, patch_idx, &mut patch_indixes);

//                 patch_noise_tmp.fill(Complex32::new(patch_noise,0.));
//                 noise_writer.write_op_from(&patch_indixes, patch_noise_tmp.as_slice_memory_order().unwrap(), accum).unwrap();
//                 norm.write_op_from(&patch_indixes, ones.as_slice_memory_order().unwrap(), accum).unwrap();

//                 for col in patch.axis_iter(Axis(1)) {
//                     writer.write_op_from(&patch_indixes, col.as_slice_memory_order().unwrap(), accum).unwrap();
//                     patch_indixes.par_iter_mut().for_each(|idx| *idx += padded_volume_stride);
//                 }
//             });
//         }
//     }
// }

#[derive(Serialize,Deserialize)]
struct Config {
    patch_planner:PatchPlanner,
    input_volumes:Vec<PathBuf>,
    volume_size:[usize;3],
    stack_size:[usize;4],
    padded_stack_size:[usize;4],
    n_processes:usize,
    batch_size:usize,
}

impl Config {
    pub fn to_file(&self,filename:impl AsRef<Path>) {
        let s = serde_json::to_string_pretty(&self)
        .expect("failed to serialize");
        let mut f = File::create(filename.as_ref().with_extension("json"))
        .expect("failed to create file");
        f.write_all(s.as_bytes()).expect("failed to write to file");
    }

    pub fn from_file(filename:impl AsRef<Path>) -> Self {
        let mut f = File::open(filename.as_ref().with_extension("json"))
        .expect("failed to open file");
        let mut s = String::new();
        f.read_to_string(&mut s).expect("failed to read file");
        serde_json::from_str(&s).expect("failed to deserialize config")
    }

}

fn mppca_plan(work_dir:impl AsRef<Path>,input_cfl_pattern:&str,n_volumes:usize,patch_size:[usize;3],patch_stride:[usize;3],n_processes:usize,batch_size:usize) {

    if !work_dir.as_ref().exists() {
        std::fs::create_dir_all(work_dir.as_ref())
        .expect("failed to create work dir");
    }

    let mut dims = vec![];
    println!("checking input files ...");
    let input_files:Vec<_> = (0..n_volumes).map(|i|{
        let filename = PathBuf::from(input_cfl_pattern.replace("*", &format!("{:01}",i)));
        let d = cfl::get_dims(&filename).expect("failed to load cfl header");
        if dims.is_empty() {
            dims = d;
        }else {
            assert_eq!(dims.as_slice(),d.as_slice(),"detected inconsistent cfl dims");
        }
        filename
    }).collect();

    let mut volume_size = [0usize;3];
    volume_size.iter_mut().zip(dims).for_each(|(vs,d)|*vs = d);

    let patch_planner = PatchPlanner::new(volume_size,patch_size,patch_stride);
    let padded_vol_size = patch_planner.padded_array_size();


    let stack_size = [
        volume_size[0],
        volume_size[1],
        volume_size[2],
        input_files.len(),
    ];

    let padded_stack_size = [
        padded_vol_size[0],
        padded_vol_size[1],
        padded_vol_size[2],
        input_files.len(),
    ];

    let c = Config {
        patch_planner,
        input_volumes: input_files,
        n_processes,
        batch_size,
        volume_size,
        stack_size,
        padded_stack_size,
    };

    c.to_file(work_dir.as_ref().join("config"));

}


fn mppca_build_4d(work_dir:impl AsRef<Path>) {

    println!("allocating 4-D stack...");
    let c = Config::from_file(work_dir.as_ref().join("config"));


    let writer = Arc::new(Mutex::new(
        cfl::CflWriter::new(work_dir.as_ref().join("cfl_4d"),&c.stack_size).expect("failed to initialize 4-D cfl")
    ));

    let volume_stride:usize = c.volume_size.iter().product();
    
    println!("filling 4-D stack...");
    c.input_volumes.par_iter().enumerate().for_each(|(vol_idx,vol_file)|{
        let volume_data = cfl::to_array(vol_file, true).expect("failed to read cfl volume");
        let volume_data_slice = volume_data.as_slice_memory_order()
        .expect("cfl volume is not in memory order");
        let starting_addr = vol_idx*volume_stride;
        println!("writing volume {} of {}",vol_idx+1,c.input_volumes.len());
        writer.lock().unwrap().write_slice(starting_addr,volume_data_slice)
        .expect("failed to write volume into 4-D stack");
    });

}

fn mppca_phase_correct(work_dir:impl AsRef<Path>) {
    let c = Config::from_file(work_dir.as_ref().join("config"));

    // look for 4-D cfl

    let reader = CflReader::new(work_dir.as_ref().join("cfl_4d"))
    .expect("failed to open cfl_4d");

    let mut writer = cfl::CflWriter::open(work_dir.as_ref().join("cfl_4d")).expect("failed to initialize 4-D cfl");

    let mut tmp_vol = ArrayD::<Complex32>::zeros(c.volume_size.as_slice().f());
    let mut pc_tmp = ArrayD::<Complex32>::zeros(c.volume_size.as_slice().f());

    // define window function for phase corrections
    let w = fourier::window_functions::HanningWindow::new(c.volume_size.as_slice());

    let volume_stride:usize = c.volume_size.iter().product();

    for i in 0..c.stack_size[3] {

        println!("filtering volume {} of {}",i+1,c.stack_size[3]);
        let starting_addr = i*volume_stride;

        // read volume data
        let volume_data = tmp_vol.as_slice_memory_order_mut().expect("failed to get memory order slice");
        reader.read_slice(starting_addr, volume_data).expect("failed to read from 4-D");

        // phase correct volume
        pc_tmp.assign(&tmp_vol);
        fourier::fftw::fftn_mut(&mut tmp_vol);
        w.apply(&mut tmp_vol);
        fourier::fftw::ifftn_mut(&mut tmp_vol);
        tmp_vol.par_mapv_inplace(|x| (x / x.abs()).conj() );
        pc_tmp *= &tmp_vol;

        // write volume to 4-D
        let volume_data = pc_tmp.as_slice_memory_order_mut().expect("failed to get memory order slice");
        writer.write_slice(starting_addr, volume_data).expect("failed to write to 4-D stack");

    }
}


fn mppca_pad(work_dir:impl AsRef<Path>) {

    let c = Config::from_file(work_dir.as_ref().join("config"));

    let cfl_input = work_dir.as_ref().join("cfl_4d");
    let cfl_padded_output = work_dir.as_ref().join("cfl_4d_padded");

    let padded_size = c.patch_planner.padded_array_size();

    let pad_amount = [
        padded_size[0] - c.volume_size[0],
        padded_size[1] - c.volume_size[1],
        padded_size[2] - c.volume_size[2]
    ];

    println!("pad amount: {:?}",pad_amount);
    // do padding

    let mut writer = CflWriter::new(cfl_padded_output,&[padded_size[0],padded_size[1],padded_size[2],c.stack_size[3]]).unwrap();

    let reader = CflReader::new(cfl_input).unwrap();

    let padded_volume_stride = padded_size.iter().product::<usize>();
    let volume_stride = c.volume_size.iter().product::<usize>();

    let mut non_padded_tmp = Array3::<Complex32>::zeros(c.volume_size.f());
    let mut padded_tmp = Array3::<Complex32>::zeros(padded_size.f());

    for vol in 0..c.stack_size[3] {

        println!("padding vol {} of {}",vol+1,c.stack_size[3]);
        // calculate offset addresses
        let starting_addr_non_padded = vol*volume_stride;
        let starting_addr_padded = vol*padded_volume_stride;

        reader.read_slice(starting_addr_non_padded, non_padded_tmp.as_slice_memory_order_mut().unwrap()).unwrap();

        // assign the non-padded volume to padded volume
        padded_tmp.slice_mut(s![0..c.volume_size[0],0..c.volume_size[1],0..c.volume_size[2]])
        .assign(&non_padded_tmp);

        for ax in 0..3 {
            if pad_amount[ax] > 0 {
                padded_tmp.slice_axis_mut(Axis(ax), Slice::from(c.volume_size[ax]..))
                .assign(
                    &non_padded_tmp.slice_axis(Axis(ax), Slice::from(0..pad_amount[ax]))
                );
            }
        }
        writer.write_slice(starting_addr_padded, padded_tmp.as_slice_memory_order().unwrap()).unwrap();
    }
}




fn mppca_denoise(work_dir:impl AsRef<Path>,process_idx:usize,partition_idx:usize) {


    let c = Config::from_file(work_dir.as_ref().join("config"));

    let padded_in = work_dir.as_ref().join("cfl_4d_padded");
    let denoised_out = work_dir.as_ref().join("denoised");
    let norm_out = work_dir.as_ref().join("norm");
    let noise_out = work_dir.as_ref().join("noise");

    let patch_ids = c.patch_planner.process_patch_ids(c.n_processes, process_idx, partition_idx);

    println!("n patches: {}",patch_ids.len());

    println!("n_patches for proceess {}: {}",process_idx,patch_ids.len());

    let reader = CflReader::new(padded_in).unwrap();
    let mut writer = CflWriter::open(denoised_out).unwrap();
    
    let mut noise_writer = CflWriter::open(noise_out).unwrap();
    let mut norm = CflWriter::open(norm_out).unwrap();

    let patch_size = c.patch_planner.patch_size();
    let n_volumes = c.input_volumes.len();
    let padded_volume_stride:usize = c.patch_planner.padded_array_size().iter().product();

    let mut patch_indixes = vec![0usize;patch_size];


    let accum = |a,b| a + b;

    let n_batches = ceiling_div(patch_ids.len(), c.batch_size);
    println!("n batches: {}",n_batches);

    for (batch_id,batch) in patch_ids.chunks(c.batch_size).enumerate() {

        let mut patch_data = Array3::<Complex32>::zeros((patch_size,n_volumes,batch.len()).f());
        let mut noise_data = Array1::from_elem(batch.len().f(),0f32);
        let mut patch_noise_tmp = Array1::<Complex32>::zeros(patch_size.f());
        let ones = Array1::<Complex32>::ones(patch_size.f());

        patch_data.axis_iter_mut(Axis(2)).zip(batch).for_each(|(mut patch,&patch_idx)|{
            // get the patch index values, shared over all volumes in data set
            c.patch_planner.linear_indices(partition_idx, patch_idx, &mut patch_indixes);

            for mut col in patch.axis_iter_mut(Axis(1)) {                    
                reader.read_into(&patch_indixes, col.as_slice_memory_order_mut().unwrap()).unwrap();
                patch_indixes.par_iter_mut().for_each(|idx| *idx += padded_volume_stride);
            }

        });

        println!("doing MPPCA denoising on batch {} of {} ...",batch_id+1,n_batches);
        singular_value_threshold_mppca(&mut patch_data, noise_data.as_slice_memory_order_mut().unwrap(), None);
        
        patch_data.axis_iter(Axis(2)).zip(batch).zip(noise_data.iter()).for_each(|((patch,&patch_idx),&patch_noise)|{
            // get the patch index values, shared over all volumes in data set
            c.patch_planner.linear_indices(partition_idx, patch_idx, &mut patch_indixes);

            patch_noise_tmp.fill(Complex32::new(patch_noise,0.));
            noise_writer.write_op_from(&patch_indixes, patch_noise_tmp.as_slice_memory_order().unwrap(), accum).unwrap();
            norm.write_op_from(&patch_indixes, ones.as_slice_memory_order().unwrap(), accum).unwrap();

            for col in patch.axis_iter(Axis(1)) {
                writer.write_op_from(&patch_indixes, col.as_slice_memory_order().unwrap(), accum).unwrap();
                patch_indixes.par_iter_mut().for_each(|idx| *idx += padded_volume_stride);
            }
        });
    }




}
