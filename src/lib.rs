pub mod patch_generator;
use std::{cell::RefCell, collections::HashMap, fs::File, io::{Read, Seek}, mem::size_of, path::Path};

use cfl::{ndarray::{Array2, Array3, Axis}, num_complex::{Complex, Complex32}};
use indicatif::ProgressStyle;
use ndarray_linalg::{c32, SVDInplace, SVD};
use patch_generator::PatchGenerator;

#[cfg(test)]
mod tests {

    use std::time::Instant;

    use cfl::{ndarray::{Array3, ArrayD, Axis, Ix2, Ix3, ShapeBuilder}, CflReader, CflWriter};
    use ndarray_linalg::{c32, SVD};
    use cfl::num_complex::Complex32;
    use crate::{ceiling_div, patch_generator::PatchGenerator, singular_value_threshold};

    #[test]
    fn it_works() {
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
        let dims = [197,120,120];

        // construct a collection of cfl volume readers called a data set
        let mut data_set = vec![];
        for i in 0..67 {
            let filename = format!("/Users/Wyatt/scratch/4D_test_data/raw/i{:02}",i);
            data_set.push(
                CflReader::new(filename).unwrap()
            )
        }

        println!("preparing output files ...");
        let mut data_set_write = vec![];
        for i in 0..67 {
            let filename = format!("/Users/Wyatt/scratch/4D_test_data/raw/i{:02}_out",i);
            data_set_write.push(
                CflWriter::new(filename,&dims).unwrap()
            )
        }

        // construct a patch generator for patch data extraction
        let patch_gen = PatchGenerator::new(dims, [10,10,10], [10,10,10]);

        // define the number of patches to extract in this batch
        let patch_batch_size = 500;


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
            singular_value_threshold(&mut patch_data, 200.);
            //println!("writing patches ...");

            patch_data.axis_iter_mut(Axis(2)).enumerate().for_each(|(idx,mut patch)|{
                // get the patch index values, shared over all volumes in data set
                let patch_indices = patch_gen.nth(idx + start_patch_id).unwrap();
                // iterate over each volume, assigning patch data to a single column
                for (mut col,vol) in patch.axis_iter_mut(Axis(1)).zip(data_set_write.iter_mut()) {
                    // write_from uses memory mapping internally to help with random indexing of volume files
                    vol.write_from(&patch_indices, col.as_slice_memory_order_mut().unwrap()).unwrap()
                }
            });

            start_patch_id += patch_batch_size;
        }

        // construct the patch data array, initialized to ones

        println!("done.");

    }
    

}


fn singular_value_threshold(patch_data:&mut Array3<Complex32>, threshold:f32) {

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

        //s.map_inplace(|val| if *val < threshold {*val = 0.});
        s.iter_mut().enumerate().for_each(|(i,val)| if i > 0 {*val = 0.});

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
    //prog_bar.finish_with_message("done");
}



pub fn ceiling_div(a:usize,b:usize) -> usize {
    (a + b - 1) / b
}
