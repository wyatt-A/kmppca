pub mod patch_generator;
use std::{cell::RefCell, collections::HashMap, fs::File, io::{Read, Seek}, mem::size_of, path::{Path, PathBuf}};

use cfl::{ndarray::{Array2, Array3, Axis, ShapeBuilder}, num_complex::{Complex, Complex32}};
use indicatif::ProgressStyle;
use ndarray_linalg::{c32, SVDInplace, SVD};
use patch_generator::PatchGenerator;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {

    use std::{fs::File, io::{Read, Write}, path::Path, ptr::write_bytes, time::Instant};

    use cfl::{ndarray::{parallel::prelude::{IntoParallelIterator, ParallelIterator}, Array3, Axis, Ix3, ShapeBuilder}, CflReader, CflWriter};
    use ndarray_linalg::{c32, SVD};
    use cfl::num_complex::Complex32;
    use serde::Serialize;
    use crate::{ceiling_div, patch_generator::{partition_patch_ids, PatchGenerator}, phase_correct_volume, singular_value_threshold_mppca, Plan};

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
        //     let cfl_in = format!("/Users/Wyatt/scratch/se_kspace_data/ksp/cr{}",i);
        //     let cfl_out = format!("/Users/Wyatt/scratch/se_kspace_data/ksp/cr_pc{}",i);
        //     let phase_op_out = format!("/Users/Wyatt/scratch/se_kspace_data/ksp/cr_phase{}",i);
        //     phase_correct_volume(cfl_in, cfl_out, phase_op_out);
        // });

        // construct a collection of cfl volume readers called a data set
        let mut data_set = vec![];
        for i in 0..67 {
            let filename = format!("/Users/Wyatt/scratch/se_kspace_data/ksp/cr_pc{}",i);
            data_set.push(
                CflReader::new(filename).unwrap()
            )
        }

        println!("preparing output files ...");
        let mut data_set_write = vec![];
        for i in 0..67 {
            let filename = format!("/Users/Wyatt/scratch/se_kspace_data/ksp/cr_d{}",i);
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
            singular_value_threshold_mppca(&mut patch_data, Some(8));
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

        let volume_dims = [256,256,256];
        let n_computers = 6;
        let n_volumes = 67;
        let patch_size = [10,10,10];
        let patch_stride = [5,5,5];
        let cfl_dir = "/privateShares/wa41/24.chdi.01.test/240725-3-1";

        // check if input files exist
        let input_files:Vec<_> = (0..n_volumes).map(|idx|{
            let p = Path::new(cfl_dir).join(format!("i{}",idx));
            if !p.exists() {
                panic!("file not found");
            }
            p
        }).collect();

        // pre-allocate output files
        let output_files:Vec<_> = (0..n_volumes).map(|idx|{
            let p = Path::new(cfl_dir).join(format!("d{}",idx));
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
        let mut concurrent_batches = vec![];
        for partition in partitioned_patches {
            // each of these partitions is safe to operate on in parallel because they do not
            // overlap
            let partition_chunk_size = ceiling_div(partition.len(), n_computers);
            let concurrent_work:Vec<_> = partition.chunks(partition_chunk_size).map( |chunk| chunk.to_vec() ).collect();
            concurrent_batches.push(concurrent_work);
        }

        println!("{}",concurrent_batches[0][0].len());
        println!("{}",concurrent_batches[0].len());
        println!("{}",concurrent_batches.len());

        // save plan
        let plan = Plan {
            plan_data: concurrent_batches,
            patch_generator,
            volume_dims,
            input_volumes: input_files,
            output_volumes: output_files,
        };

        let bytes = bincode::serialize(&plan).expect("failed to serialize plan");

        let mut f = File::create("plan").unwrap();
        f.write_all(&bytes).unwrap();

    }
    
    //cargo test --package kmppca --lib -- tests::test_launch --exact --nocapture
    #[test]
    fn test_launch() {

        // we are given the computer index and batch from the array job
        // and the
        let computer_idx = 0;
        let batch_index = 0;
        let n_volumes = 67;
        let batch_size = 2000;

        let mut f = File::open("plan").unwrap();
        let mut byte_buff = vec![];
        f.read_to_end(&mut byte_buff).unwrap();
        let plan: Plan = bincode::deserialize(&byte_buff).unwrap();

        println!("plan len: {}",plan.plan_data.len());

        let patches_to_process = &plan.plan_data[batch_index][computer_idx];
        let n_patches_to_process = patches_to_process.len();

        let n_batches = ceiling_div(n_patches_to_process, batch_size);

        println!("n_batches = {}",n_batches);
        println!("n total patches = {}",n_patches_to_process);
        
        let mut writers:Vec<_> = plan.output_volumes.iter().map(|path| CflWriter::open(path).unwrap()).collect();
        let readers:Vec<_> = plan.input_volumes.iter().map(|path| CflReader::new(path).unwrap()).collect();

        for batch in patches_to_process.chunks(batch_size) {
            let mut patch_data = Array3::from_elem((plan.patch_generator.patch_size(),n_volumes,batch.len()).f(), Complex32::ZERO);
            patch_data.axis_iter_mut(Axis(2)).enumerate().for_each(|(idx,mut patch)|{
                // get the patch index values, shared over all volumes in data set
                let patch_indices = plan.patch_generator.nth(batch[idx]).unwrap();
                // iterate over each volume, assigning patch data to a single column
                for (mut col,vol) in patch.axis_iter_mut(Axis(1)).zip(readers.iter()) {
                    // read_into uses memory mapping internally to help with random indexing of volume files
                    vol.read_into(&patch_indices, col.as_slice_memory_order_mut().unwrap()).unwrap()
                }
            });

            singular_value_threshold_mppca(&mut patch_data, Some(8));

            let write_operation = |a,b| a + b;

            patch_data.axis_iter_mut(Axis(2)).enumerate().for_each(|(idx,mut patch)|{
                // get the patch index values, shared over all volumes in data set
                let patch_indices = plan.patch_generator.nth(batch[idx]).unwrap();
                // iterate over each volume, assigning patch data to a single column
                for (mut col,vol) in patch.axis_iter_mut(Axis(1)).zip(writers.iter_mut()) {
                    // write_from uses memory mapping internally to help with random indexing of volume files
                    vol.write_op_from(&patch_indices, col.as_slice_memory_order_mut().unwrap(),write_operation).unwrap()
                }
            });

        }

    }



}

#[derive(Serialize,Deserialize)]
struct Plan {
    patch_generator:PatchGenerator,
    plan_data:Vec<Vec<Vec<usize>>>,
    volume_dims:[usize;3],
    input_volumes:Vec<PathBuf>,
    output_volumes:Vec<PathBuf>,
}

impl Plan {
    pub fn index(&self,batch:usize,computer:usize,patch:usize) -> usize {
        self.plan_data[batch][computer][patch]
    }
}


fn singular_value_threshold_mppca(patch_data:&mut Array3<Complex32>, rank:Option<usize>) {

    let m = patch_data.shape()[0];
    let n = patch_data.shape()[1];

    let mut _s = Array2::from_elem((m,n), Complex32::ZERO);
    
    let prog_bar = indicatif::ProgressBar::new(patch_data.shape()[2] as u64);
    prog_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}").unwrap()
            .progress_chars("=>-")
    );
    
    patch_data.axis_iter_mut(Axis(2)).for_each(|mut matrix| {
        let (u,mut s,v) = matrix.svd(true, true).unwrap();

        let thresh = if let Some(rank) = rank {
            rank
        }else {
            let (t_idx,_) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);
            t_idx
        };
        
        s.iter_mut().enumerate().for_each(|(i,val)| if i >= thresh {*val = 0.});

        let u = u.unwrap();
        let v = v.unwrap();
        let mut diag_view = _s.diag_mut();
        diag_view.assign(
            &s.map(|x|Complex32::new(*x,0.))
        );
        let denoised_matrix = u.dot(&_s).dot(&v);
        matrix.assign(&denoised_matrix);
        prog_bar.inc(1);
    });
    prog_bar.finish();
}

pub fn ceiling_div(a:usize,b:usize) -> usize {
    (a + b - 1) / b
}

fn phase_correct_volume<P:AsRef<Path>>(cfl_in:P, cfl_out:P, phase_op_out:P) {
    let x = cfl::to_array(cfl_in, true).unwrap();
    let w = fft::window_functions::HanningWindow::new(x.shape());
    let (pc,phase) = image_utils::unwrap_phase::phase_correct(&x,Some(&w));
    cfl::from_array(cfl_out, &pc).unwrap();
    cfl::from_array(phase_op_out, &phase).unwrap();
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

/// returns the index of the singular value to threshold, along with the estimated matrix variance of the matrix
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