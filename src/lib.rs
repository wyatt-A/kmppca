pub mod patch_generator;
use std::{cell::RefCell, collections::HashMap, fs::File, io::{Read, Seek, Write}, mem::size_of, path::{Path, PathBuf}, sync::{Arc, Mutex}};
use cfl::{ndarray::{parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator}, Array2, Array3, ArrayD, Axis, ShapeBuilder}, num_complex::{Complex, Complex32, ComplexFloat}};
use fourier::window_functions::WindowFunction;
use indicatif::ProgressStyle;
use ndarray_linalg::{c32, SVDInplace, SVDInto, SVD, SVDDC};
use patch_generator::{partition_patch_ids, PatchGenerator};
use serde::{Deserialize, Serialize};
use cfl::ndarray::parallel::prelude::ParallelIterator;
pub mod patch_planner;
use cfl::ndarray::parallel::prelude::*;
pub mod slurm;

#[cfg(test)]
mod tests {

    use std::{fs::File, io::{Read, Write}, path::Path, time::Instant};

    use cfl::{ndarray::{parallel::prelude::{IntoParallelIterator, ParallelIterator}, Array1, Array3, Axis, Ix3, ShapeBuilder}, CflReader, CflWriter};
    use ndarray_linalg::SVD;
    use cfl::num_complex::Complex32;
    use crate::{ceiling_div, patch_generator::{partition_patch_ids, PatchGenerator}, singular_value_threshold_mppca, Plan};

    #[test]
    fn test_svd_decomp_recon() {
        let x = cfl::to_array("/Users/Wyatt/scratch/4D_test_data/pc/i00", true).unwrap().into_dimensionality::<Ix3>().unwrap();
        let y = x.index_axis(Axis(0), 0);
        let (u,s,v) = y.svd(true, true).unwrap();
        let s = s.diag();
        println!("{:?}",s);
        let r = u.unwrap() * s * v.unwrap().t();
    }

    //cargo test --release --package kmppca --lib -- tests::vol_reader --exact --nocapture
    #[test]
    fn vol_reader() {

        let now = Instant::now();
        let dims = [10,480,480];

        println!("calculating phase operators ...");

        // (0..67).into_par_iter().for_each(|i|{
        //     println!("working on {} of 67 ...",i+1);
        //     let cfl_in = format!("/home/wyatt/test_data/kspace/c{}",i);
        //     let cfl_out = format!("/home/wyatt/test_data/kspace/pc{}",i);
        //     let phase_op_out = format!("/home/wyatt/test_data/kspace/po{}",i);
        //     phase_correct_volume(cfl_in, cfl_out, phase_op_out);
        // });

        // construct a collection of cfl volume readers called a data set
        let mut data_set = vec![];
        for i in 0..67 {
            let filename = format!("/home/wyatt/test_data/kspace/pc{}",i);
            data_set.push(
                CflReader::new(filename).unwrap()
            )
        }

        println!("preparing output files ...");
        let mut data_set_write = vec![];
        for i in 0..67 {
            let filename = format!("/home/wyatt/test_data/kspace/cd{}",i);
            data_set_write.push(
                CflWriter::new(filename,&dims).unwrap()
                //CflWriter::open(filename).unwrap()
            )
        }

        // construct a patch generator for patch data extraction
        let patch_gen = PatchGenerator::new(dims, [10,10,10], [10,1,1]);

        // define the number of patches to extract in this batch
        let patch_batch_size = 2000;

        let n_batches = patch_gen.n_patches() / patch_batch_size;
        let remainder = patch_gen.n_patches() % patch_batch_size;

        let mut start_patch_id = 0;

        for batch in 0..n_batches {
            println!("processing batch {} of {} ...",batch+1,n_batches);
            let mut patch_data = Array3::from_elem((patch_gen.patch_size(),data_set.len(),patch_batch_size).f(), Complex32::ZERO);
            let mut noise_data = Array1::from_elem(patch_batch_size,0f32);


            //println!("reading patches ...");
            // iterate over the patch batch
            patch_data.axis_iter_mut(Axis(2)).enumerate().for_each(|(idx,mut patch)|{
                // get the patch index values, shared over all volumes in data set
                let patch_indices = patch_gen.nth(idx + start_patch_id).unwrap();
                // iterate over each volume, assigning patch data to a single column
                for (mut col,vol) in patch.axis_iter_mut(Axis(1)).zip(data_set.iter_mut()) {
                    // read_into uses memory mapping internally to help with random indexing of volume files
                    vol.read_into(&patch_indices, col.as_slice_memory_order_mut().unwrap()).unwrap()
                }
            });

            //println!("processing patches ...");
            //singular_value_threshold_mppca(&mut patch_data, None);
            singular_value_threshold_mppca(&mut patch_data, noise_data.as_slice_memory_order_mut().unwrap(), Some(8));
            //println!("writing patches ...");

            // integrate the values in the file
            let write_operation = |a,b| a + b;

            patch_data.axis_iter_mut(Axis(2)).enumerate().for_each(|(idx,mut patch)|{
                // get the patch index values, shared over all volumes in data set
                let patch_indices = patch_gen.nth(idx + start_patch_id).unwrap();
                // iterate over each volume, assigning patch data to a single column
                for (mut col,vol) in patch.axis_iter_mut(Axis(1)).zip(data_set_write.iter_mut()) {
                    // write_from uses memory mapping internally to help with random indexing of volume files
                    vol.write_op_from(&patch_indices, col.as_slice_memory_order_mut().unwrap(),write_operation).unwrap()
                }
            });

            start_patch_id += patch_batch_size;
        }

        // construct the patch data array, initialized to ones

        let dur = now.elapsed();
        println!("done.");
        println!("took {} sec",dur.as_secs_f64());
    
    }

    //cargo test --package kmppca --lib -- tests::test_denoise --exact --nocapture
    #[test]
    fn test_denoise() {

        let volume_dims = [788,480,480];
        let n_computers = 1;
        let n_volumes = 67;
        let patch_size = [20,20,20];
        let patch_stride = [20,20,20];
        let cfl_dir = "/privateShares/wa41/24.chdi.01.test/240725-3-1";

        // check if input files exist
        let input_files:Vec<_> = (0..n_volumes).map(|idx|{
            let p = Path::new(cfl_dir).join(format!("i{}.cfl",idx));
            if !p.exists() {
                panic!("file not found {:?}",p);
            }
            p
        }).collect();

        println!("pre-allocating output files ...");
        // pre-allocate output files
        let output_files:Vec<_> = (0..n_volumes).into_par_iter().map(|idx|{
            let p = Path::new(cfl_dir).join(format!("d{}",idx));
            println!("pre-allocating {:?}",p);
            CflWriter::new(&p,&volume_dims).unwrap();
            p
        }).collect();

        // create a patch generator that prescribes how to read and write the data set
        let patch_generator = PatchGenerator::new(volume_dims, patch_size, patch_stride);

        let n_total_patches = patch_generator.n_patches();
        println!("n total patches: {}",n_total_patches);

        // partion the patch ids into groups on non-overlapping patches
        println!("partitioning patch ids ...");
        let partitioned_patches = partition_patch_ids(&patch_generator);

        // we will probably want to cache this data because it is largly static
        let n_partitions = partitioned_patches.len();
        println!("n partitions: {}",n_partitions);

        // build the plan
        let mut partitioned_patch_chunks = vec![];
        for partition in partitioned_patches {
            // each of these partitions is safe to operate on in parallel because they do not
            // overlap
            let partition_chunk_size = ceiling_div(partition.len(), n_computers);
            let concurrent_work:Vec<_> = partition.chunks(partition_chunk_size).map( |chunk| chunk.to_vec() ).collect();
            partitioned_patch_chunks.push(concurrent_work);
        }

        println!("{}",partitioned_patch_chunks[0][0].len());
        println!("{}",partitioned_patch_chunks[0].len());
        println!("{}",partitioned_patch_chunks.len());

        // save plan
        let plan = Plan {
            plan_data: partitioned_patch_chunks,
            patch_generator,
            volume_dims,
            n_volumes,
            n_processes: n_computers,
        };

        let bytes = bincode::serialize(&plan).expect("failed to serialize plan");
        let mut f = File::create("plan").unwrap();
        f.write_all(&bytes).unwrap();

    }

}

#[derive(Serialize,Deserialize)]
pub struct Plan {
    pub patch_generator:PatchGenerator,
    pub plan_data:Vec<Vec<Vec<usize>>>,
    pub volume_dims:[usize;3],
    pub n_volumes:usize,
    pub n_processes:usize,
}

impl Plan {
    pub fn new(volume_dims:[usize;3],n_volumes:usize,patch_size:[usize;3],patch_stride:[usize;3],n_processes:usize) -> Self {
        let patch_generator = PatchGenerator::new(volume_dims, patch_size, patch_stride);

        let n_total_patches = patch_generator.n_patches();
        println!("n total patches: {}",n_total_patches);

        // partion the patch ids into groups of non-overlapping patches
        println!("partitioning patch ids ...");
        let partitioned_patches = partition_patch_ids(&patch_generator);

        // we will probably want to cache this data because it is largly static
        let n_partitions = partitioned_patches.len();
        println!("n partitions: {}",n_partitions);

        // build the plan
        let mut partitioned_patch_chunks = vec![];
        for partition in partitioned_patches {
            // each of these partitions is safe to operate on in parallel because they do not
            // overlap
            let partition_chunk_size = ceiling_div(partition.len(), n_processes);
            let concurrent_work:Vec<_> = partition.chunks(partition_chunk_size).map( |chunk| chunk.to_vec() ).collect();
            partitioned_patch_chunks.push(concurrent_work);
        }
        
        Self {
            plan_data: partitioned_patch_chunks,
            patch_generator,
            volume_dims,
            n_volumes,
            n_processes,
        }

    }

    pub fn to_file(&self,filename:impl AsRef<Path>) {
        let bytes = bincode::serialize(self).expect("failed to serialize plan");
        println!("writing to plan file {:?}",filename.as_ref());
        let mut f = File::create(filename).expect("failed to create plan file");
        f.write_all(&bytes).expect("failed to write to file");
    }

    pub fn from_file(filename:impl AsRef<Path>) -> Self {
        let mut f = File::open(filename).expect("failed to open file");
        let mut bytes = vec![];
        f.read_to_end(&mut bytes).expect("failed to read file");
        bincode::deserialize(&bytes).expect("failed to deserialize plan file")
    }

}

#[derive(Serialize,Deserialize)]
pub struct Config {
    pub patch_plan_file:PathBuf,
    pub input_files:Vec<PathBuf>,
    pub output_files:Vec<PathBuf>,
    pub noise_volume:PathBuf,
    pub normalization_volume:PathBuf,
    pub batch_size:usize,
}

impl Config {

    pub fn to_file(&self,filename:impl AsRef<Path>) {
        let bytes = bincode::serialize(self).expect("failed to serialize plan");
        println!("writing to config file {:?}",filename.as_ref());
        let mut f = File::create(filename).expect("failed to create config file");
        f.write_all(&bytes).expect("failed to write to file");
    }

    pub fn from_file(filename:impl AsRef<Path>) -> Self {
        let mut f = File::open(filename).expect("failed to open file");
        let mut bytes = vec![];
        f.read_to_end(&mut bytes).expect("failed to read file");
        bincode::deserialize(&bytes).expect("failed to deserialize config file")
    }
}

pub fn _singular_value_threshold_mppca(matrix:&mut Array2<Complex32>, rank:Option<usize>) -> Option<f32> {

        let m = matrix.shape()[0];
        let n = matrix.shape()[1];

        //let mut _s = Array2::from_elem((m,n), Complex32::ZERO);

        let nn = m.min(n);
        let mut _s = Array2::from_elem((nn,nn), Complex32::ZERO);

        //let (_,s,_) = matrix.svd(true, true).unwrap();
        // let (_,sigma_sq) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);
        // matrix.par_mapv_inplace(|x| x / sigma_sq);
        //let (u,mut s,v) = matrix.svd(true, true).unwrap();

        let (u,mut s,v) = matrix.svddc(ndarray_linalg::UVTFlag::Some).unwrap();

        let u = u.unwrap();
        let v = v.unwrap();

        // println!("u:{:?}",u.shape());
        // println!("s:{:?}",s.shape());
        // println!("v:{:?}",v.shape());

        let mut noise = None;

        let rank = rank.unwrap_or_else(||{
            let (rank,sigma_sq) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);
            noise = Some(sigma_sq);
            rank
        });
        
        s.iter_mut().enumerate().for_each(|(i,val)| if i >= rank {*val = 0.});

        //let u = u.unwrap();
        //let v = v.unwrap();
        let mut diag_view = _s.diag_mut();
        diag_view.assign(
            &s.map(|x|Complex32::new(*x,0.))
        );

        // if u in 1x1
        let denoised_matrix = if u.len() == 1 {
            _s.dot(&v)
        }else {
            u.dot(&_s).dot(&v)
        };

        //let denoised_matrix = u.dot(&_s).dot(&v);
        matrix.assign(&denoised_matrix);
        noise
}

pub fn singular_value_threshold_mppca(patch_data:&mut Array3<Complex32>, noise:&mut [f32], rank:Option<usize>) {

    let m = patch_data.shape()[0];
    let n = patch_data.shape()[1];

    
    // let prog_bar = indicatif::ProgressBar::new(patch_data.shape()[2] as u64);
    // prog_bar.set_style(
    //     ProgressStyle::default_bar()
    //         .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}").unwrap()
    //         .progress_chars("=>-")
    // );

    //let count = Arc::new(Mutex::new(0usize));
    

    patch_data.axis_iter_mut(Axis(2)).into_par_iter().zip(noise.par_iter_mut()).for_each(|(mut matrix,noise)| {
    //patch_data.axis_iter_mut(Axis(2)).zip(noise.iter_mut()).for_each(|(mut matrix,noise)| {

        let mut _s = Array2::from_elem((m,n), Complex32::ZERO);

        //let (_,s,_) = matrix.svd(true, true).unwrap();
        // let (_,sigma_sq) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);
        // matrix.par_mapv_inplace(|x| x / sigma_sq);
        let (u,mut s,v) = matrix.svd(true, true).unwrap();
        let (rank,sigma_sq) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);

        *noise = sigma_sq;
        // let thresh = if let Some(rank) = rank {
        //     rank
        // }else {
        //     let (t_idx,sigma_sq) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);

        //     matrix.par_mapv_inplace(|x| x / sigma_sq);

        //     t_idx
        // };
        
        s.iter_mut().enumerate().for_each(|(i,val)| if i >= rank {*val = 0.});

        let u = u.unwrap();
        let v = v.unwrap();
        let mut diag_view = _s.diag_mut();
        diag_view.assign(
            &s.map(|x|Complex32::new(*x,0.))
        );
        let denoised_matrix = u.dot(&_s).dot(&v);
        //denoised_matrix.par_mapv_inplace(|x| x * sigma_sq);
        matrix.assign(&denoised_matrix);
        //let mut g = count.lock().unwrap();  
        //*g += 1;
        //println!("{}",g);
        //prog_bar.inc(1);
    });
    //prog_bar.finish();
}

pub fn ceiling_div(a:usize,b:usize) -> usize {
    (a + b - 1) / b
}

// fn phase_correct_volume<P:AsRef<Path>>(cfl_in:P, cfl_out:P, phase_op_out:P) {
//     let x = cfl::to_array(cfl_in, true).unwrap();
//     let w = fourier::window_functions::HanningWindow::new(x.shape());
//     let (pc,phase) = phase_correct(&x,Some(&w));
//     cfl::from_array(cfl_out, &pc).unwrap();
//     cfl::from_array(phase_op_out, &phase).unwrap();
// }

pub fn phase_correct<W:WindowFunction>(complex_img:&ArrayD<Complex32>, window_function:Option<&W>) -> (ArrayD<Complex32>,ArrayD<Complex32>) {
    let mut tmp = complex_img.clone();
    if let Some(w) = window_function {
        fourier::rustfft::fftn(&mut tmp);
        w.apply(&mut tmp);
        fourier::rustfft::ifftn(&mut tmp);
    }
    let phase_correction = tmp.map(|x|x.unscale(x.abs()).conj());
    (complex_img * &phase_correction, phase_correction)
}

/*
function x_denoised = denoise_mppca(x)
    [U,S,V]=svd(x,'econ');
    S=diag(S);
    MM=size(x,1);
    NNN=size(x,2);
    R = min(MM, NNN);
    scaling = (max(MM, NNN) - (0:R-1)) / NNN;
    scaling = scaling(:);
    vals=S;
    vals = (vals).^2 / NNN;
    % First estimation of Sigma^2;  Eq 1 from ISMRM presentation
    csum = cumsum(vals(R:-1:1));
    cmean = csum(R:-1:1)./(R:-1:1)';
    sigmasq_1 = cmean./scaling;
    % Second estimation of Sigma^2; Eq 2 from ISMRM presentation
    gamma = (MM - (0:R-1)) / NNN;
    rangeMP = 4*sqrt(gamma(:));
    rangeData = vals(1:R) - vals(R);
    sigmasq_2 = rangeData./rangeMP;
    t = find(sigmasq_2 < sigmasq_1, 1);
    %sigmasq_2(t)
    energy_scrub=sqrt(sum(S.^1)).\sqrt(sum(S(t:end).^1));
    S(t:end)=0;
    x_denoised = U * diag(S) * V';
*/

/// returns the index of the singular value to threshold, along with the estimated matrix variance
fn marchenko_pastur_singular_value(singular_values:&[f32],m:usize,n:usize) -> (usize,f32) {
    
    let r = m.min(n);
    let mut vals = singular_values.to_owned();

    let scaling:Vec<_> = (0..r).clone().map(|x| (m.max(n) as f32 - x as f32) / n as f32).collect();
    
    vals.iter_mut().for_each(|x| *x = x.powi(2) / n as f32);
    vals.reverse();
    let mut csum = cumsum(&vals);
    csum.reverse();
    vals.reverse();

    let cmean:Vec<_> = csum.iter().zip((1..r+1).rev()).map(|(x,y)| x / y as f32).collect();
    let sigmasq_1:Vec<_> = cmean.iter().zip(scaling.iter()).map(|(x,y)| *x / *y).collect();
    let range_mp:Vec<_> = (0..r).map(|x| x as f32).map(|x| (m as f32 - x) /  n as f32).map(|x| x.sqrt() * 4.).collect();
    let range_data:Vec<_> = vals[0..r].iter().map(|x| x - vals[r-1]).collect();
    let sigmasq_2:Vec<_> = range_data.into_iter().zip(range_mp).map(|(x,y)|x / y).collect();

    //println!("{:?}",sigmasq_1);
    //println!("{:?}",sigmasq_2);

    let idx = sigmasq_1.iter().zip(sigmasq_2).enumerate().find_map(|(i,(s1,s2))|{
        if s2 < *s1 {
            Some(i)
        }else {
            None
        }
    }).unwrap();

    let variance_estimate = sigmasq_1[idx];

    (idx,variance_estimate)
}


fn cumsum(x:&[f32]) -> Vec<f32> {
    let mut x = x.to_owned();
    let mut s = 0.;
    x.iter_mut().for_each(|val|{
        *val += s;
        s = *val;
    });
    x
}

#[test]
fn cumsum_test() {
    let x = cumsum(&[1.,2.,3.,4.]);
    println!("{:?}",x);
}

#[test]
fn mp_test() {

    let matrix_entries:Vec<f32> = vec![
        -0.1337,2.0024,0.8491,2.1531,
        0.4786,0.2538,0.5708,-0.2315
    ];

    let matrix = Array2::from_shape_vec((4usize,2usize).f(), matrix_entries).unwrap();
    println!("{:?}",matrix);
    let (u,s,vt) = matrix.svd(true,true).unwrap();

    let s = s.to_vec();

    let (i,var) = marchenko_pastur_singular_value(&s, 4, 2);

    println!("var: {}",var);


}