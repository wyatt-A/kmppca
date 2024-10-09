use std::{fmt::write, fs::File, io::{Read, Write}, ops::Range, path::{Path, PathBuf}, process::Command, sync::{Arc, Mutex}, time::Instant};
use cfl::{
    ndarray::{
        parallel::prelude::{
            IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator}, s, Array1, Array3, ArrayD, Axis, ShapeBuilder, Zip}, num_complex::ComplexFloat, CflReader, CflWriter};
use fourier::window_functions::WindowFunction;
use kmppca::{ceiling_div, patch_planner::PatchPlanner, singular_value_threshold_mppca, slurm::{self, SlurmTask}};
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
    DenoisePartition(DenoisePartitionArgs),
    AllocateOutput(AllocateOutputArgs),
    Reconstruct(ReconstructArgs),
}

#[derive(Parser, Debug, Clone)]
struct AllocateOutputArgs {
    config:PathBuf,
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
    config:PathBuf,
}

#[derive(Parser, Debug, Clone)]
struct DenoiseArgs {
    config:PathBuf,
    #[clap(short = 'p')]
    partition_idx:Option<usize>
}

#[derive(Parser, Debug, Clone)]
struct DenoisePartitionArgs {
    config:PathBuf,
    process_id:usize,
    partition_id:usize,
}

#[derive(Parser, Debug, Clone)]
struct ReconstructArgs {
    config:PathBuf,
    partition_id:usize,
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
            mppca_build_4d(&init_args.config);
            mppca_phase_correct(&init_args.config);
            mppca_pad(&init_args.config);
        },
        SubCmd::Denoise(denoise_args) => {
            let c = Config::from_file(&denoise_args.config);
            mppca_allocate_output(&denoise_args.config);
            if slurm::is_installed() {
                launch_array_jobs(&denoise_args.config,"denoise_test",80_000,denoise_args.partition_idx); // 2 gigs of memory per process
            }else {
                for partition in 0..c.patch_ranges.len() {
                    for process in 0..c.patch_ranges[partition].len() {
                        mppca_denoise(&denoise_args.config, process, partition);
                    }
                }
            }
        },
        SubCmd::DenoisePartition(denoise_partition_args) => {
            mppca_denoise(
                denoise_partition_args.config,
                denoise_partition_args.process_id,
                denoise_partition_args.partition_id
            );
        },
        SubCmd::AllocateOutput(allocate_output_args) => {
            mppca_allocate_output(allocate_output_args.config)
        },
        SubCmd::Reconstruct(reconstruct_args) => {
            mppca_reconstruct(reconstruct_args.config,reconstruct_args.partition_id);
        }
    }
}

fn launch_array_jobs(config:impl AsRef<Path>,job_name:&str,req_mem_mb:usize,partition_idx:Option<usize>) {

    let c = Config::from_file(&config);

    let this_exe = std::env::current_exe().expect("failed to get this exe");

    let mut dep_job:Option<u64> = None;

    // single partition mode
    if let Some(p_idx) = partition_idx {

        let ranges = c.patch_ranges.get(p_idx).unwrap();

        // set up call to denoise-partition with the SLURM_ARRAY_TASK_ID variable
        let mut cmd = Command::new(&this_exe);
        cmd.arg("denoise-partition");
        cmd.arg(&config.as_ref());
        cmd.arg("$SLURM_ARRAY_TASK_ID");
        cmd.arg(p_idx.to_string());

        // set up the array jobs over n_processes (length of patch id ranges)
        let mut task = SlurmTask::new(&c.work_dir,&format!("{}_partition_{}",job_name,p_idx),req_mem_mb)
        .output(&c.work_dir)
        .array(0, ranges.len()-1)
        .command(cmd);

        // launch partition array jobs with "ok" dependencies
        let jid = if let Some(&dep_job_id) = dep_job.as_ref() {
            task.job_dependency_after_ok(dep_job_id)
            .submit()
        }else {
            task.submit()
        };

        let mut recon_cmd = Command::new(&this_exe);
        recon_cmd.arg("reconstruct");
        recon_cmd.arg(&config.as_ref());
        recon_cmd.arg(p_idx.to_string());

        let mut recon_task = SlurmTask::new(&c.work_dir,&format!("{}_reconstruct_{}",job_name,p_idx),req_mem_mb)
        .output(&c.work_dir)
        .job_dependency_after_ok(jid)
        //.on_node("n018")
        .command(recon_cmd);
        recon_task.submit();

        return
    };


    for (partition,ranges) in c.patch_ranges.iter().enumerate() {
        
        // set up call to denoise-partition with the SLURM_ARRAY_TASK_ID variable
        let mut cmd = Command::new(&this_exe);
        cmd.arg("denoise-partition");
        cmd.arg(&config.as_ref());
        cmd.arg("$SLURM_ARRAY_TASK_ID");
        cmd.arg(partition.to_string());

        // set up the array jobs over n_processes (length of patch id ranges)
        let mut task = SlurmTask::new(&c.work_dir,&format!("{}_partition_{}",job_name,partition),req_mem_mb)
        .output(&c.work_dir)
        .array(0, ranges.len()-1)
        .command(cmd);

        // launch partition array jobs with "ok" dependencies
        let jid = if let Some(&dep_job_id) = dep_job.as_ref() {
            task.job_dependency_after_ok(dep_job_id)
            .submit()
        }else {
            task.submit()
        };

        let mut recon_cmd = Command::new(&this_exe);
        recon_cmd.arg("reconstruct");
        recon_cmd.arg(&config.as_ref());
        recon_cmd.arg(partition.to_string());

        let mut recon_task = SlurmTask::new(&c.work_dir,&format!("{}_reconstruct_{}",job_name,partition),req_mem_mb)
        .output(&c.work_dir)
        .job_dependency_after_ok(jid)
        //.on_node("n018")
        .command(recon_cmd);
        let jid = recon_task.submit();

        dep_job = Some(jid);
    }

}

#[derive(Clone,Copy)]
pub enum ResultFile {
    Denoised,
    Padded,
    NonPadded,
    Norm,
    Noise,
    OutputVolume{index:usize}
}

#[derive(Serialize,Deserialize)]
struct Config {
    work_dir:PathBuf,
    patch_planner:PatchPlanner,
    input_volumes:Vec<PathBuf>,
    volume_size:[usize;3],
    stack_size:[usize;4],
    padded_stack_size:[usize;4],
    n_processes:usize,
    batch_size:usize,
    patch_ranges:Vec<Vec<Range<usize>>>,
}

impl Config {

    pub fn result_filename(&self,result_file:ResultFile) -> PathBuf {
        match result_file {
            ResultFile::Denoised => self.work_dir.join("denoised"),
            ResultFile::Padded => self.work_dir.join("padded"),
            ResultFile::NonPadded => self.work_dir.join("non_padded"),
            ResultFile::Norm => self.work_dir.join("norm"),
            ResultFile::Noise => self.work_dir.join("noise"),
            ResultFile::OutputVolume { index } => {
                let n = self.input_volumes.len(); // Number of things to enumerate
                let width = ((n - 1) as f64).log10().ceil() as usize;
                let filename = format!("d{:0width$}", index, width = width);
                self.work_dir.join(filename)
            },
        }
    }

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

fn mppca_allocate_output(config:impl AsRef<Path>) {
    let c = Config::from_file(config);
    let denoise_out = c.result_filename(ResultFile::Denoised);
    let noise_out = c.result_filename(ResultFile::Noise);
    let norm_out = c.result_filename(ResultFile::Norm);
    let a = CflWriter::new(denoise_out,&c.padded_stack_size).unwrap();
    let b = CflWriter::new(noise_out,&c.patch_planner.padded_array_size()).unwrap();
    let c = CflWriter::new(norm_out,&c.patch_planner.padded_array_size()).unwrap();

    a.flush().unwrap();
    b.flush().unwrap();
    c.flush().unwrap();

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

    let n_partitions = patch_planner.n_total_partitions();
    
    // organized by partition in the outer dimension, and process in the inner dimension
    let mut patch_ranges = vec![];
    for partition_idx in 0..n_partitions {
        let n = patch_planner.n_total_partition_patches_lin(partition_idx);
        patch_ranges.push(
            // splits the patch indices into at most n groups (case where n_process is greater than n_patches)
            split_indices(n, n_processes)
        )
    }

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
        work_dir:work_dir.as_ref().to_path_buf(),
        patch_planner,
        input_volumes: input_files,
        n_processes,
        batch_size,
        volume_size,
        stack_size,
        padded_stack_size,
        patch_ranges,
    };

    c.to_file(work_dir.as_ref().join("config"));

}


fn mppca_build_4d(config_file:impl AsRef<Path>) {

    println!("allocating 4-D stack...");
    let c = Config::from_file(config_file);


    let writer = Arc::new(Mutex::new(
        cfl::CflWriter::new(c.result_filename(ResultFile::NonPadded),&c.stack_size).expect("failed to initialize 4-D cfl")
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

    println!("flushing cfl buffer ...");
    writer.lock().unwrap().flush().expect("failed to flush writer");

}

fn mppca_phase_correct(config_file:impl AsRef<Path>) {
    let c = Config::from_file(config_file);

    // look for 4-D cfl

    let reader = CflReader::new(c.result_filename(ResultFile::NonPadded))
    .expect("failed to open cfl_4d");

    let mut writer = cfl::CflWriter::open(c.result_filename(ResultFile::NonPadded)).expect("failed to initialize 4-D cfl");

    let mut tmp_vol = ArrayD::<Complex32>::zeros(c.volume_size.as_slice().f());
    let mut pc_tmp = ArrayD::<Complex32>::zeros(c.volume_size.as_slice().f());

    // define window function for phase corrections
    let w = fourier::window_functions::HanningWindow::new(c.volume_size.as_slice()).window(c.volume_size.as_slice());

    let volume_stride:usize = c.volume_size.iter().product();

    for i in 0..c.stack_size[3] {

        println!("phase correcting volume {} of {}",i+1,c.stack_size[3]);
        let starting_addr = i*volume_stride;

        // read volume data
        let volume_data = tmp_vol.as_slice_memory_order_mut().expect("failed to get memory order slice");
        reader.read_slice(starting_addr, volume_data).expect("failed to read from 4-D");

        // phase correct volume
        pc_tmp.assign(&tmp_vol);
        fourier::fftw::fftn_mut(&mut tmp_vol);
        // apply window function
        Zip::from(&mut tmp_vol).and(&w).par_for_each(|x,&w|{
            *x = w * *x;
        });
        fourier::fftw::ifftn_mut(&mut tmp_vol);
        tmp_vol.par_mapv_inplace(|x| (x / x.abs()).conj() );
        pc_tmp *= &tmp_vol;

        // write volume to 4-D
        let volume_data = pc_tmp.as_slice_memory_order_mut().expect("failed to get memory order slice");
        writer.write_slice(starting_addr, volume_data).expect("failed to write to 4-D stack");

    }

    println!("flushing cfl buffer ...");
    writer.flush().expect("failed to flush writer");
}


fn mppca_pad(config_file:impl AsRef<Path>) {

    let c = Config::from_file(config_file);

    let padded_size = c.patch_planner.padded_array_size();

    let pad_amount = [
        padded_size[0] - c.volume_size[0],
        padded_size[1] - c.volume_size[1],
        padded_size[2] - c.volume_size[2]
    ];

    println!("pad amount: {:?}",pad_amount);
    // do padding

    let mut writer = CflWriter::new(c.result_filename(ResultFile::Padded),&[padded_size[0],padded_size[1],padded_size[2],c.stack_size[3]]).unwrap();

    let reader = CflReader::new(c.result_filename(ResultFile::NonPadded)).unwrap();

    let padded_volume_stride = padded_size.iter().product::<usize>();
    let volume_stride = c.volume_size.iter().product::<usize>();

    //let mut non_padded_tmp = Array3::<Complex32>::zeros(c.volume_size.f());
    //let mut padded_tmp = Array3::<Complex32>::zeros(padded_size.f());

    for vol in 0..c.stack_size[3] {

        println!("padding vol {} of {}",vol+1,c.stack_size[3]);
        // calculate offset addresses
        let starting_addr_non_padded = vol*volume_stride;
        let starting_addr_padded = vol*padded_volume_stride;

        let mut non_padded_tmp = Array3::<Complex32>::zeros(c.volume_size.f());
        reader.read_slice(starting_addr_non_padded, non_padded_tmp.as_slice_memory_order_mut().unwrap()).unwrap();
    
        // Initialize padded array with full padded dimensions
        let mut padded_tmp = Array3::<Complex32>::zeros(padded_size.f());
        
        padded_tmp.slice_mut(s![0..c.volume_size[0],0..c.volume_size[1],0..c.volume_size[2]])
        .assign(
            &non_padded_tmp
        );

        non_padded_tmp = padded_tmp.clone();

        if pad_amount[0] > 0 {
            padded_tmp.slice_mut(s![c.volume_size[0]..,..,..])
            .assign(
                &non_padded_tmp.slice(s![0..pad_amount[0],..,..])
            );
            //non_padded_tmp = padded_tmp.clone();
            non_padded_tmp.assign(&padded_tmp);
        }

        if pad_amount[1] > 0 {
            padded_tmp.slice_mut(s![..,c.volume_size[1]..,..])
            .assign(
                &non_padded_tmp.slice(s![..,0..pad_amount[1],..])
            );
            //non_padded_tmp = padded_tmp.clone();
            non_padded_tmp.assign(&padded_tmp);
        }

        if pad_amount[2] > 0 {
            padded_tmp.slice_mut(s![..,..,c.volume_size[2]..])
            .assign(
                &non_padded_tmp.slice(s![..,..,0..pad_amount[2]])
            );
        }

        writer.write_slice(starting_addr_padded, padded_tmp.as_slice_memory_order().unwrap()).unwrap();
    }

    writer.flush().expect("failed to flush writer");
}


fn mppca_denoise(config_file:impl AsRef<Path>,process_idx:usize,partition_idx:usize) {

    let c = Config::from_file(config_file);

    let padded_in = c.result_filename(ResultFile::Padded);
    // let denoised_out = c.result_filename(ResultFile::Denoised);
    // let norm_out = c.result_filename(ResultFile::Norm);
    // let noise_out = c.result_filename(ResultFile::Noise);

    let patch_range = c.patch_ranges.get(partition_idx)
    .expect("partition index out of range")
    .get(process_idx)
    .expect("process index out-of-range")
    .clone();

    let patch_ids:Vec<_> = patch_range.collect();

    // let patch_ids = c.patch_planner.process_patch_ids(c.n_processes, process_idx, partition_idx);

    println!("n patches: {}",patch_ids.len());

    println!("n_patches for proceess {}: {}",process_idx,patch_ids.len());

    let reader = CflReader::new(padded_in).unwrap();
    // let mut writer = CflBufWriter::open(denoised_out).unwrap();
    
    // let mut noise_writer = CflBufWriter::open(noise_out).unwrap();
    // let mut norm = CflBufWriter::open(norm_out).unwrap();

    

    let patch_size = c.patch_planner.patch_size();
    let n_volumes = c.input_volumes.len();
    let padded_volume_stride:usize = c.patch_planner.padded_array_size().iter().product();

    let mut patch_indixes = vec![0usize;patch_size];

    let mut denoised_patch_file = CflWriter::new(c.work_dir.join(format!("pd_{}_{}",process_idx,partition_idx)),&[patch_size,n_volumes,patch_ids.len()]).unwrap();
    let mut denoised_patch_write_address = 0;

    let mut noise_file = CflWriter::new(c.work_dir.join(format!("noise_{}_{}",process_idx,partition_idx)),&[patch_ids.len()]).unwrap();
    let mut noise_write_address = 0;

    //let accum = |a,b| a + b;

    let n_batches = ceiling_div(patch_ids.len(), c.batch_size);
    println!("n batches: {}",n_batches);

    for (batch_id,batch) in patch_ids.chunks(c.batch_size).enumerate() {

        let mut patch_data = Array3::<Complex32>::zeros((patch_size,n_volumes,batch.len()).f());
        let mut noise_data = Array1::from_elem(batch.len().f(),0f32);
        // let mut patch_noise_tmp = Array1::<Complex32>::zeros(patch_size.f());
        // let ones = Array1::<Complex32>::ones(patch_size.f());

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
        println!("batch complete");
        
        // instead of writing back to these files directly, we need to write to our own file
        // to the later aggrigate with a single process. This is to avoid NFS + multiple computer
        // problems & read/write corruption

        let slice = patch_data.as_slice_memory_order().unwrap();
        denoised_patch_file.write_slice(denoised_patch_write_address, slice).unwrap();
        denoised_patch_write_address += slice.len();

        let nd = noise_data.map(|&x|Complex32::new(x,0.));
        let slice = nd.as_slice_memory_order().unwrap();
        noise_file.write_slice(noise_write_address, slice).unwrap();
        noise_write_address += slice.len();


        // patch_data.axis_iter(Axis(2)).zip(batch).zip(noise_data.iter()).for_each(|((patch,&patch_idx),&patch_noise)|{

        //     println!("writing patch {}",patch_idx);
        //     // get the patch index values, shared over all volumes in data set
        //     c.patch_planner.linear_indices(partition_idx, patch_idx, &mut patch_indixes);

        //     patch_noise_tmp.fill(Complex32::new(patch_noise,0.));
        //     noise_writer.write_op_from(&patch_indixes, patch_noise_tmp.as_slice_memory_order().unwrap(), accum).unwrap();
        //     norm.write_op_from(&patch_indixes, ones.as_slice_memory_order().unwrap(), accum).unwrap();

        //     for col in patch.axis_iter(Axis(1)) {
        //         writer.write_op_from(&patch_indixes, col.as_slice_memory_order().unwrap(), accum).unwrap();
        //         patch_indixes.par_iter_mut().for_each(|idx| *idx += padded_volume_stride);
        //     }
        // });



    }

    //writer.flush().expect("failed to flush writer");
    denoised_patch_file.flush().expect("failed to flush writer");
    noise_file.flush().expect("failed to flush writer");

}

fn mppca_reconstruct(config:impl AsRef<Path>,partition_idx:usize) {
    let c = Config::from_file(config);

    // let denoise_file = c.result_filename(ResultFile::Denoised);
    // let noise_file = c.result_filename(ResultFile::Noise);
    // let norm_file = c.result_filename(ResultFile::Norm);

    // std::fs::rename(&denoise_file.with_extension("hdr"), &denoise_file.with_file_name("denoised_r").with_extension("hdr")).unwrap();
    // std::fs::rename(&denoise_file.with_extension("cfl"), &denoise_file.with_file_name("denoised_r").with_extension("cfl")).unwrap();

    // std::fs::rename(&noise_file.with_extension("hdr"), &noise_file.with_file_name("noise_r").with_extension("hdr")).unwrap();
    // std::fs::rename(&noise_file.with_extension("cfl"), &noise_file.with_file_name("noise_r").with_extension("cfl")).unwrap();

    // std::fs::rename(&norm_file.with_extension("hdr"), &norm_file.with_file_name("norm_r").with_extension("hdr")).unwrap();
    // std::fs::rename(&norm_file.with_extension("cfl"), &norm_file.with_file_name("norm_r").with_extension("cfl")).unwrap();


    // let reader = CflReader::new(&denoise_file.with_file_name("denoised_r")).unwrap();
    // let noise_reader = CflReader::new(noise_file.with_file_name("noise_r")).unwrap();
    // let norm_reader = CflReader::new(norm_file.with_file_name("norm_r")).unwrap();

    let writer = Arc::new(Mutex::new(
        CflWriter::open(c.result_filename(ResultFile::Denoised)).unwrap()
    ));

    let noise_writer = Arc::new(Mutex::new(
        CflWriter::open(c.result_filename(ResultFile::Noise)).unwrap()
    ));

    let norm_writer = Arc::new(Mutex::new(
        CflWriter::open(c.result_filename(ResultFile::Norm)).unwrap()
    ));

    let patch_size = c.patch_planner.patch_size();
    let n_volumes = c.stack_size[3];

    let padded_volume_stride = c.patch_planner.padded_array_size().iter().product::<usize>();

    let patch_ranges = &c.patch_ranges.get(partition_idx).unwrap().clone();
    //let mut patch_indices = vec![0usize;patch_size];
    let ones = vec![Complex32::ONE;patch_size];
    //let mut norm_tmp = vec![Complex32::ZERO;patch_size];

    let write_operation = |a,b| a + b;

    patch_ranges.par_iter().enumerate().for_each(|(process_idx,process)|{
    //for (process_idx,process) in patch_ranges.iter().enumerate() {


        let mut patch_indices = vec![0usize;patch_size];
        // let mut norm_tmp = vec![Complex32::ZERO;patch_size];
        // let mut noise_tmp = Array1::from_elem(patch_size.f(), Complex32::ZERO);

        println!("working on process {}",process_idx);

        let denoised_patch_file = CflReader::new(c.work_dir.join(format!("pd_{}_{}",process_idx,partition_idx))).unwrap();
        let mut denoised_patch_read_address = 0;

        let noise_data_file = CflReader::new(c.work_dir.join(format!("noise_{}_{}",process_idx,partition_idx))).unwrap();
        let mut noise_data_read_address = 0;

        let mut patch_noise_tmp = Array1::<Complex32>::zeros(patch_size.f());
        //let mut patch_noise_read_tmp = Array1::<Complex32>::zeros(patch_size.f());

        let process_patch_ids = process.clone().collect::<Vec<_>>();

        for (batch_id,batch) in process_patch_ids.chunks(c.batch_size).enumerate() {

            let mut patch_data = Array3::<Complex32>::zeros((patch_size,n_volumes,batch.len()).f());
            let mut noise_data = Array1::<Complex32>::zeros(batch.len().f());


            let slice = patch_data.as_slice_memory_order_mut().unwrap();
            denoised_patch_file.read_slice(denoised_patch_read_address, slice).unwrap();
            denoised_patch_read_address += slice.len();

            let slice = noise_data.as_slice_memory_order_mut().unwrap();
            noise_data_file.read_slice(noise_data_read_address, slice).unwrap();
            noise_data_read_address += slice.len();


            patch_data.axis_iter(Axis(2)).zip(batch.iter().zip(noise_data)).for_each(|(patch,(&patch_idx,patch_noise))|{
                // get the patch index values, shared over all volumes in data set
                c.patch_planner.linear_indices(partition_idx, patch_idx, &mut patch_indices);

                patch_noise_tmp.fill(patch_noise);
                //noise_reader.read_into(&patch_indices, &mut patch_noise_read_tmp.as_slice_memory_order_mut().unwrap()).unwrap();
                //patch_noise_tmp += &patch_noise_read_tmp;
                noise_writer.lock().unwrap().write_op_from(&patch_indices, patch_noise_tmp.as_slice_memory_order().unwrap(),write_operation).unwrap();

                //norm_reader.read_into(&patch_indices, &mut norm_tmp).unwrap();
                //norm_tmp.iter_mut().for_each(|x| *x += Complex32::ONE);
                norm_writer.lock().unwrap().write_op_from(&patch_indices, &ones, write_operation).unwrap();

                for col in patch.axis_iter(Axis(1)) {
                    //println!("writing patch ...");
                    //reader.read_into(&patch_indices, noise_tmp.as_slice_memory_order_mut().unwrap()).unwrap();
                    //noise_tmp += &col;
                    writer.lock().unwrap().write_op_from(&patch_indices, col.as_slice_memory_order().unwrap(),write_operation).unwrap();
                    patch_indices.par_iter_mut().for_each(|idx| *idx += padded_volume_stride);
                }
            });
        }
    });

    writer.lock().unwrap().flush().expect("failed to flush writer");
    noise_writer.lock().unwrap().flush().expect("failed to flush writer");
    norm_writer.lock().unwrap().flush().expect("failed to flush writer");

    // std::fs::remove_file(&denoise_file.with_file_name("denoised_r").with_extension("hdr")).unwrap();
    // std::fs::remove_file(&denoise_file.with_file_name("denoised_r").with_extension("cfl")).unwrap();
    // std::fs::remove_file(&noise_file.with_file_name("noise_r").with_extension("hdr")).unwrap();
    // std::fs::remove_file(&noise_file.with_file_name("noise_r").with_extension("cfl")).unwrap();
    // std::fs::remove_file(&norm_file.with_file_name("norm_r").with_extension("hdr")).unwrap();
    // std::fs::remove_file(&norm_file.with_file_name("norm_r").with_extension("cfl")).unwrap();

}


fn mppca_split_4d(config:impl AsRef<Path>) {

    let c = Config::from_file(config);

    let denoised_reader = CflReader::new(c.result_filename(ResultFile::Denoised)).unwrap();

    let norm = cfl::to_array(c.result_filename(ResultFile::Norm), true).unwrap();

    let padded_volume_stride = c.patch_planner.padded_array_size().iter().product::<usize>();

    let mut padded_temp = Array3::from_elem(c.patch_planner.padded_array_size().f(), Complex32::ZERO);
    let mut unpadded_temp = ArrayD::from_elem(c.volume_size.as_slice().f(), Complex32::ZERO);

    for vol in 0..c.stack_size[3] {
        let address = vol * padded_volume_stride;
        denoised_reader.read_slice(address, padded_temp.as_slice_memory_order_mut().unwrap()).unwrap();
        // normalize denoised volume by norm volume
        padded_temp /= &norm;
        unpadded_temp.assign(
            &padded_temp.slice(s![0..c.volume_size[0],0..c.volume_size[1],0..c.volume_size[2]])
        );

        let out = c.result_filename(ResultFile::OutputVolume { index: vol });
        cfl::from_array(out, &unpadded_temp).unwrap();
    }

}

fn split_indices(n: usize, batches: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::with_capacity(batches);

    if batches >= n {
        for i in 0..n {
            ranges.push(
                i..(i+1)
            )
        }
        return ranges;
    }

    let mut ranges = Vec::with_capacity(batches);
    let batch_size = n / batches;
    let remainder = n % batches;
    
    let mut start = 0;
    
    for i in 0..batches {
        let end = start + batch_size + if i == batches - 1 { remainder } else { 0 };
        ranges.push(start..end);
        start = end;
    }
    
    ranges
}

#[test]
fn test_idx_split() {

    let n = 11;
    let n_processes = 5;

    let r = split_indices(n, n_processes);

    println!("{:?}",r);


}