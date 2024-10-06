use std::{fmt::write, fs::File, io::Write, path::{Path, PathBuf}, sync::{Arc, Mutex}, time::Instant};

use bincode::de::read;
use cfl::{ndarray::{parallel::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator}, s, Array1, Array3, ArrayD, Axis, ShapeBuilder, Slice}, num_complex::ComplexFloat, CflReader, CflWriter};
use fourier::window_functions::WindowFunction;
use kmppca::{ceiling_div, patch_planner::PatchPlanner, singular_value_threshold_mppca};
use ndarray_linalg;
use cfl::num_complex::Complex32;
use serde::{Deserialize, Serialize};


fn main() {

    // 4-D data set of complex images over q-space
    // do phase-corrections

    let inputs = "/home/wyatt/test_data/raw";
    let n_volumes = 67;

    let dims = [197,120,120,n_volumes];

    println!("allocating 4-D stack...");
    let mut writer = cfl::CflWriter::new("test4d",&dims).unwrap();

    let volume_stride:usize = dims[0..3].iter().product();
    
    for i in 0..n_volumes {
        let cfl_in = Path::new(inputs).join(format!("i{:02}",i));
        let volume_data = cfl::to_array(cfl_in, true).expect("failed to read cfl volume");
        let volume_data_slice = volume_data.as_slice_memory_order()
        .expect("cfl volume is not in memory order");
        let starting_addr = i*volume_stride;
        println!("writing volume {} of {}",i+1,n_volumes);
        writer.write_slice(starting_addr,volume_data_slice).expect("failed to write volume into 4-D stack");
    }
    


    // do phase corrections

    let reader = cfl::CflReader::new("test4d").expect("failed to create 4-D cfl reader");

    let mut tmp_vol = ArrayD::<Complex32>::zeros(dims[0..3].f());
    let mut pc_tmp = ArrayD::<Complex32>::zeros(dims[0..3].f());

    // define window function for phase corrections
    let w = fourier::window_functions::HanningWindow::new(&dims[0..3]);

    for i in 0..n_volumes {

        println!("filtering volume {} of {}",i+1,n_volumes);
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

    let vol_size = [dims[0],dims[1],dims[2]];
    let patch_planner = PatchPlanner::new(vol_size,[10,10,10],[10,10,10]);
    let padded_size = patch_planner.padded_array_size();
    let pad_amount = [padded_size[0] - vol_size[0],padded_size[1] - vol_size[1],padded_size[2] - vol_size[2]];

    println!("pad amount: {:?}",pad_amount);
    // do padding

    let writer = Arc::new(Mutex::new(
        CflWriter::new("padded",&[padded_size[0],padded_size[1],padded_size[2],n_volumes]).unwrap()
    ));

    let reader = CflReader::new("test4d").unwrap();

    let padded_volume_stride = padded_size.iter().product::<usize>();

    let mut non_padded_tmp = Array3::<Complex32>::zeros(vol_size.f());
    let mut padded_tmp = Array3::<Complex32>::zeros(padded_size.f());

    for vol in 0..n_volumes {

        println!("padding vol {} of {}",vol+1,n_volumes);
        // calculate offset addresses
        let starting_addr_non_padded = vol*volume_stride;
        let starting_addr_padded = vol*padded_volume_stride;

        reader.read_slice(starting_addr_non_padded, non_padded_tmp.as_slice_memory_order_mut().unwrap()).unwrap();

        // assign the non-padded volume to padded volume
        padded_tmp.slice_mut(s![0..vol_size[0],0..vol_size[1],0..vol_size[2]])
        .assign(&non_padded_tmp);

        for ax in 0..3 {
            if pad_amount[ax] > 0 {
                padded_tmp.slice_axis_mut(Axis(ax), Slice::from(vol_size[ax]..))
                .assign(
                    &non_padded_tmp.slice_axis(Axis(ax), Slice::from(0..pad_amount[ax]))
                );
            }
        }

        writer.lock().unwrap().write_slice(starting_addr_padded, padded_tmp.as_slice_memory_order().unwrap()).unwrap();
    }


    // do mppca denoising

    let reader = CflReader::new("padded").unwrap();
    let mut writer = CflWriter::new("output",&[padded_size[0],padded_size[1],padded_size[2],n_volumes]).unwrap();
    
    let mut noise_writer = CflWriter::new("noise",&[padded_size[0],padded_size[1],padded_size[2]]).unwrap();
    let mut norm = CflWriter::new("norm",&[padded_size[0],padded_size[1],padded_size[2]]).unwrap();

    let batch_size = 1000;

    let mut patch_indixes = vec![0usize;patch_planner.patch_size()];


    let accum = |a,b| a + b;
    //let noise_write_op = |_,b| b;


    let n_partitions = patch_planner.n_total_partitions();
    for partition in 0..n_partitions {
        println!("working on partition {} of {}..",partition+1,n_partitions);

        let n_patches = patch_planner.n_total_partition_patches_lin(partition);

        let patch_ids:Vec<_> = (0..n_patches).collect();

        let n_batches = ceiling_div(n_patches, batch_size);

        for (batch_id,batch) in patch_ids.chunks(batch_size).enumerate() {

            let mut patch_data = Array3::<Complex32>::zeros((patch_planner.patch_size(),n_volumes,batch.len()).f());
            let mut noise_data = Array1::from_elem(batch.len().f(),0f32);
            let mut patch_noise_tmp = Array1::<Complex32>::zeros(patch_planner.patch_size().f());
            let ones = Array1::<Complex32>::ones(patch_planner.patch_size().f());

            patch_data.axis_iter_mut(Axis(2)).zip(batch).for_each(|(mut patch,&patch_idx)|{
                // get the patch index values, shared over all volumes in data set
                patch_planner.linear_indices(partition, patch_idx, &mut patch_indixes);

                for mut col in patch.axis_iter_mut(Axis(1)) {                    
                    reader.read_into(&patch_indixes, col.as_slice_memory_order_mut().unwrap()).unwrap();
                    patch_indixes.par_iter_mut().for_each(|idx| *idx += padded_volume_stride);
                }

            });

            println!("doing MPPCA denoising on batch {} of {} ...",batch_id+1,n_batches);
            singular_value_threshold_mppca(&mut patch_data, noise_data.as_slice_memory_order_mut().unwrap(), None);
            
            patch_data.axis_iter(Axis(2)).zip(batch).zip(noise_data.iter()).for_each(|((patch,&patch_idx),&patch_noise)|{
                // get the patch index values, shared over all volumes in data set
                patch_planner.linear_indices(partition, patch_idx, &mut patch_indixes);

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
}

#[derive(Serialize,Deserialize)]
struct Config {
    patch_planner:PatchPlanner,
    input_volumes:Vec<PathBuf>,
    n_processes:usize,
}

impl Config {
    pub fn to_file(filename:&str)
}

fn mppca_plan(work_dir:impl AsRef<Path>,input_cfl_pattern:&str,n_volumes:usize,patch_size:[usize;3],patch_stride:[usize;3],n_processes:usize) {

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

    let c = Config {
        patch_planner,
        input_volumes: input_files,
        n_processes,
    };

    let config_string = serde_json::to_string_pretty(&c).expect("failed to serialize config");
    let mut f = File::create(work_dir.as_ref().join("config").with_extension("json")).expect("failed to write config");
    f.write_all(config_string.as_bytes()).expect("failed to write to config file");

}

fn mppca_launch(work_dir:impl AsRef<Path>,process_idx:usize) {

    let config = work_dir.as_ref().join("config").with_extension("toml");




}