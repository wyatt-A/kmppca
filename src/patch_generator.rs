use std::collections::HashSet;

use indicatif::ProgressStyle;
use serde::{Deserialize, Serialize};

use crate::ceiling_div;

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use crate::patch_generator::PatchGenerator;
    use cfl::ndarray::{ArrayD,ShapeBuilder};
    use cfl::num_complex::Complex32;
    use indicatif::ProgressStyle;
    use rand::Rng;

    use super::partition_patch_ids;

    //cargo test --release --package kmppca --lib -- patch_generator::tests::patch_decomposition_reconstruction--exact --nocapture
    #[test]
    fn patch_decomposition_reconstruction() {

        // create a random array of values
        let dims = [197,120,120];
        // generates patches of size 10,10,10 with stride of 5,5,5 for a patch overlap of 5
        let patch_generator = PatchGenerator::new(dims,[10,10,10],[5,5,5]);

        println!("generating random data volume with size {:?}",dims);
        let data_size = dims.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let test_volume_data:Vec<_> = (0..data_size).map(|_| {
            Complex32::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0)
            )
        }).collect();

        let test_volume = ArrayD::from_shape_vec(dims.as_slice().f(), test_volume_data).unwrap();
        // this just points to the underlying fortran-ordered data
        let volume_data = test_volume.as_slice_memory_order().unwrap();

        let n_patches = patch_generator.n_patches();
        println!("iterating over {} patches",n_patches);

        let mut patch_matrix = vec![];
        println!("extracting patches ...");
        for patch in &patch_generator {
            let mut patch_data = vec![];
            patch.into_iter().for_each(|idx|{
                patch_data.push(
                    volume_data[idx]
                )
            });
            patch_matrix.push(patch_data);
        }

        // create a destination array to compare to the original after reconstruction
        let mut y = volume_data.to_owned();
        y.fill(Complex32::ZERO);

        println!("reconstructing from patches ...");
        for (patch_idx,patch) in patch_generator.iter().enumerate() {
            let data = &patch_matrix[patch_idx];
            data.iter().zip(patch.into_iter()).for_each(|(val,idx)|{
                y[idx] = *val;
            })
        }

        println!("checking consistency ...");
        assert_eq!(y.as_slice(),volume_data)
        
    }

    #[test]
    fn non_overlapping_patches() {
        let patch_generator = PatchGenerator::new([4,4,4],[2,2,2],[2,2,2]);
        let partitions = partition_patch_ids(&patch_generator);
        for partition in partitions {
            println!("{:?}",partition)
        }
    }
}

/// partitions patch ids such that each partion is garaunteed to operate
/// over the image volume in a non-overlapping manner. This is useful for splitting
/// work over multiple concurrent processes without having to deal with race conditions
pub fn partition_patch_ids(patch_generator:&PatchGenerator) -> Vec<Vec<usize>> {

    let mut partitioned_patch_ids:Vec<Vec<usize>> = vec![vec![]];
    let mut partition_entries:Vec<HashSet<usize>> = vec![HashSet::new()];

    let prog_bar = indicatif::ProgressBar::new(patch_generator.n_patches() as u64);
    prog_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}").unwrap()
            .progress_chars("=>-")
    );
    
    for (patch_index,patch_entries) in patch_generator.iter().enumerate() {
        let mut patch_entries = HashSet::from_iter(patch_entries.into_iter());
        // try to find a partition for the patch, draining the patch if one is found
        for (partition_entry,patch_ids) in partition_entries.iter_mut().zip(partitioned_patch_ids.iter_mut()) {
            if partition_entry.is_disjoint(&patch_entries) {
                partition_entry.extend(patch_entries.drain());
                patch_ids.push(patch_index);
                break
            }
        };
        // if the patch has not been drained because no suitable partition was found,
        // create a new partition
        if !patch_entries.is_empty() {
            partition_entries.push(
                HashSet::from_iter(patch_entries.into_iter())
            );
            partitioned_patch_ids.push(vec![patch_index]);
        }

        if patch_index % 100 == 0 {
            prog_bar.inc(100);
        }
        
    }
    prog_bar.finish();

    partitioned_patch_ids
}

#[derive(Serialize,Deserialize)]
pub struct PatchGenerator {
    x: CircularWindow1D,
    y: CircularWindow1D,
    z: CircularWindow1D,
}

impl<'a> IntoIterator for &'a PatchGenerator {
    type Item = Vec<usize>;

    type IntoIter = PatchIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        PatchIterator {
            x: &self.x,
            y: &self.y,
            z: &self.z,
            x_state: 0,
            y_state: 0,
            z_state: 0,
            is_exhausted: false,
        }
    }
}


pub struct PatchIterator<'a> {
    x: &'a CircularWindow1D,
    y: &'a CircularWindow1D,
    z: &'a CircularWindow1D,
    x_state:usize,
    y_state:usize,
    z_state:usize,
    is_exhausted:bool
}


impl<'a> PatchIterator<'a> {
    fn advance_state(&mut self) -> bool {

        // Increment x
        self.x_state += 1;

        // Handle the carry over for y and z
        if self.x_state == self.x.n_iter() {
            self.x_state = 0;
            self.y_state += 1;
        }

        if self.y_state == self.y.n_iter() {
            self.y_state = 0;
            self.z_state += 1;
        }

        // Check if z has reached its maximum
        if self.z_state == self.z.n_iter() {
            return false; // Indicates the end of iteration
        }

        true // Continue iteration
        
    }

    fn nth(&mut self,idx:usize) -> Option<Vec<usize>> {
        self.x_state = 0;
        self.y_state = 0;
        self.z_state = 0;
        for _ in 0..idx {
            if !self.advance_state() {
                return None
            }
        }
        Some(self.patch_indices())
    }

    fn patch_indices(&self) -> Vec<usize> {
        let x = self.x.nth(self.x_state).unwrap();
        let y = self.y.nth(self.y_state).unwrap();
        let z = self.z.nth(self.z_state).unwrap();
        let patch_size = x.len() * y.len() * z.len();
        let mut idx = Vec::<usize>::with_capacity(patch_size);
        for i in &x {
            for j in &y {
                for k in &z {
                    idx.push(
                        array_utils::sub_to_idx_col_major(
                            &[*i,*j,*k],
                            &[self.x.array_len,self.y.array_len,self.z.array_len]
                        ).unwrap()
                    );
                }
            }
        }
        idx
    }
}

impl<'a> Iterator for PatchIterator<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_exhausted {
            return None
        }else {
            let indices = self.patch_indices();
            if !self.advance_state() {
                self.is_exhausted = true;
            }
            return Some(indices)
        }
    }
}


impl PatchGenerator {

    /// construct a new patch generator over an array
    pub fn new(array_size:[usize;3],patch_size:[usize;3],strides:[usize;3]) -> Self {
        array_size.iter().zip(patch_size.iter()).for_each(|(x,p)|{
            assert!(x >= p, "patch dimensions must be less than or equal to array dimensions")
        });
        array_size.iter().zip(strides.iter()).for_each(|(x,p)|{
            assert!(x >= p, "stride sizes must be less than or equal to array dimensions")
        });
        Self {
            x: CircularWindow1D::new(array_size[0],patch_size[0],strides[0]),
            y: CircularWindow1D::new(array_size[1],patch_size[1],strides[1]),
            z: CircularWindow1D::new(array_size[2],patch_size[2],strides[2]),
        }
    }

    /// returns the nth patch index values
    pub fn nth(&self,patch_idx:usize) -> Option<Vec<usize>> {
        self.iter().nth(patch_idx)
    }

    /// returns the number of entries of a patch
    pub fn patch_size(&self) -> usize {
        self.x.window_len * self.y.window_len * self.z.window_len
    }

    /// returns the total number of patches to generate
    pub fn n_patches(&self) -> usize {
        self.x.n_iter() * self.y.n_iter() * self.z.n_iter()
    }

    /// return an iterator over all patches
    pub fn iter(&self) -> PatchIterator {
        self.into_iter()
    }

}

#[derive(Debug,Clone,Serialize,Deserialize)]
struct CircularWindow1D {
    array_len:usize,
    window_len: usize,
    stride: usize,
}

impl CircularWindow1D {

    fn new(array_len:usize,window_len:usize,stride:usize) -> Self {
        Self {
            array_len,
            window_len,
            stride,
        }
    }
    fn nth(&self,stride_idx:usize) -> Option<Vec<usize>> {
        let mut buffer = vec![0;self.window_len];
        if stride_idx < self.n_iter() {
            let start = stride_idx * self.stride;
            buffer.iter_mut()
                .zip((start..start + self.window_len).map(|i| i % self.array_len))
                .for_each(|(dest, src)| *dest = src);
        
            Some(buffer)
        }else {
            None
        }
    }

    fn n_iter(&self) -> usize {
        ceiling_div(self.array_len, self.stride)
    }
}

impl CircularWindow1D {
    pub fn iter(&self) -> CircularWindowIterator {
        CircularWindowIterator {
            window: &self,
            stride_index: 0,
            n_iter: ceiling_div(self.array_len, self.stride),
        }
    }
}

#[derive(Debug,Clone)]
struct CircularWindowIterator<'a> {
    window: &'a CircularWindow1D,
    stride_index: usize,
    n_iter:usize,
}

impl<'a> CircularWindowIterator<'a> {
    fn reset(&mut self) {
        self.stride_index = 0;
    }


}

impl<'a> Iterator for CircularWindowIterator<'a> {
    type Item = Vec<usize>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(buff) = self.window.nth(self.stride_index) {
            self.stride_index += 1;
            return Some(buff)
        }else {
            None
        }
    }
}