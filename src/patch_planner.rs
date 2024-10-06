use std::{ops::Range, time::Instant};
use cfl::num_complex::Complex32;

use crate::ceiling_div;

pub struct PatchPlanner {
    x:Patch1D,
    y:Patch1D,
    z:Patch1D,
}

impl PatchPlanner {
    pub fn new(array_size:[usize;3],patch_size:[usize;3],patch_stride:[usize;3]) -> Self {
        Self {
            x: Patch1D::new(array_size[0],patch_size[0],patch_stride[0]),
            y: Patch1D::new(array_size[1],patch_size[1],patch_stride[1]),
            z: Patch1D::new(array_size[2],patch_size[2],patch_stride[2]),
        }
    }

    pub fn patch_size(&self) -> usize {
        self.x.patch_size * self.y.patch_size * self.z.patch_size
    }

    fn n_partitions(&self) -> [usize;3] {
        [
            self.x.n_partitions(),
            self.y.n_partitions(),
            self.z.n_partitions()
        ]
    }

    pub fn n_total_partitions(&self) -> usize {
        self.n_partitions().into_iter().product()
    }

    pub fn n_patches(&self) -> [usize;3] {
        [
            self.x.n_total_patches(),
            self.y.n_total_patches(),
            self.z.n_total_patches(),
        ]
    }

    fn n_partition_patches(&self,partition_idx:[usize;3]) -> [usize;3] {
        [
            self.x.n_partition_patches(partition_idx[0]),
            self.y.n_partition_patches(partition_idx[1]),
            self.z.n_partition_patches(partition_idx[2]),
        ]
    }

    fn n_partition_patches_lin(&self,partition_idx:usize) -> [usize;3] {
        let shape = self.n_partitions();
        let mut sub = [0usize;3];
        array_utils::idx_to_sub_col_major(partition_idx, &shape, &mut sub).unwrap();
        self.n_partition_patches(sub)
    }

    pub fn n_total_partition_patches_lin(&self,partition_idx:usize) -> usize {
        self.n_partition_patches_lin(partition_idx).into_iter().product()
    }

    pub fn padded_array_size(&self) -> [usize;3] {
        [
            self.x.padded_array_size(),
            self.y.padded_array_size(),
            self.z.padded_array_size(),
        ]
    }

    fn address_range(&self,partition_idx:[usize;3],patch_idx:[usize;3]) -> [Range<usize>;3] {
        [
            self.x.address_range(partition_idx[0], patch_idx[0]),
            self.y.address_range(partition_idx[1], patch_idx[1]),
            self.z.address_range(partition_idx[2], patch_idx[2]),
        ]
    }

    pub fn linear_indices(&self,partition_idx:usize,patch_idx:usize,indices:&mut [usize]) {
        assert_eq!(indices.len(),self.patch_size());
        let mut partition_sub = [0;3];
        let mut patch_sub = [0;3];
        let n_partitions = self.n_partitions();
        array_utils::idx_to_sub_col_major(partition_idx, &n_partitions, &mut partition_sub).unwrap();
        let n_patches = self.n_partition_patches(partition_sub);
        array_utils::idx_to_sub_col_major(patch_idx, &n_patches, &mut patch_sub).unwrap();
        self._linear_indices(partition_sub, patch_sub, indices);
    }

    fn _linear_indices(&self, partition_idx: [usize; 3], patch_idx: [usize; 3], indices: &mut [usize]) {
        assert_eq!(indices.len(), self.patch_size());
    
        let [x, y, z] = self.address_range(partition_idx, patch_idx);
        let shape = self.padded_array_size();
    
        let mut idx = 0;
    
        // Compute the strides for column-major order.
        let x_stride = 1;
        let y_stride = shape[0];        // Stride for `j`
        let z_stride = shape[0] * shape[1]; // Stride for `k`
    
        for k in z {
            let k_base = k * z_stride; // Precompute part of the index
            for j in y.clone() {
                let jk_base = k_base + j * y_stride; // Combine `j` and `k` contributions
                for i in x.clone() {
                    // Instead of calling an external function, we calculate inline
                    indices[idx] = jk_base + i * x_stride;
                    idx += 1;
                }
            }
        }
    }
}


//cargo test --package kmppca --lib -- patch_planner::test --exact --nocapture
#[test]
fn test() {

    let p = PatchPlanner::new([788,480,480],[10,10,10],[8,8,8]);

    let mut patch_indices = vec![0usize;p.patch_size()];
    let data = vec![Complex32::ONE;p.patch_size()];

    let padded_size = p.padded_array_size();

    let write_op = |a,b| a + b;
    //let mut test_array1 = cfl::CflWriter::new("test1",&padded_size).unwrap();
    let mut test_array = cfl::CflWriter::new("test1",&padded_size).unwrap();

    for part in 0..p.n_total_partitions() {
        for patch in 0..p.n_total_partition_patches_lin(part) {
            p.linear_indices(part, patch, &mut patch_indices);
            test_array.write_op_from(&patch_indices, &data, write_op).unwrap();
        }
    }

}


struct Patch1D {
    array_size:usize,
    patch_size:usize,
    patch_stride:usize,
}

impl Patch1D {
    pub fn new(array_size:usize,patch_size:usize,patch_stride:usize) -> Self {

        assert!(array_size > 0);
        assert!(patch_size > 0);
        assert!(patch_stride > 0);
        assert!(patch_size <= array_size);

        Patch1D {
            array_size,
            patch_size,
            patch_stride,
        }
    }

    pub fn n_partitions(&self) -> usize {
        let n_partitions = ceiling_div(self.patch_size, self.patch_stride);
        //println!("number of sub-problems: {}",n_partitions);
        n_partitions
    }

    pub fn n_partition_patches(&self,partition_idx:usize) -> usize {
        let n_partitions = self.n_partitions();
        assert!(partition_idx < n_partitions);
        ceiling_div(
            self.array_size - partition_idx*self.patch_stride,
            n_partitions * self.patch_stride
        )
    }

    pub fn n_total_patches(&self) -> usize{
        ceiling_div(self.array_size,self.patch_stride)
    }

    pub fn address_range(&self,partition_idx:usize,patch_idx:usize) -> Range<usize> {
        let n_partitions = ceiling_div(self.patch_size, self.patch_stride);
        assert!(partition_idx < n_partitions);
        let n_partition_patches = self.n_partition_patches(partition_idx);
        assert!(patch_idx < n_partition_patches);
        let partition_start = partition_idx*self.patch_stride;
        let start_address = partition_start + patch_idx * n_partitions*self.patch_stride;
        start_address..(start_address + self.patch_size)
    }

    pub fn padded_array_size(&self) -> usize {
        let gamma = ceiling_div(self.array_size,self.patch_stride);
        (gamma - 1) * self.patch_stride + self.patch_size
    }

}

#[test]
fn test3() {
    let p = Patch1D::new(197,197,197);
    println!("{:?}",p.address_range(0, 0));
    println!("padded: {}",p.padded_array_size());
}



#[test]
fn test2() {

    let array_size = 197usize;
    let patch_size = 10usize;
    let patch_stride = 8usize;

    println!("array size: {}",array_size);
    println!("patch size: {}",patch_size);
    println!("patch stride: {}",patch_stride);

    // this represents the number of 1-partitions and the partition stride
    let beta = ceiling_div(patch_size, patch_stride);
    println!("number of sub-problems: {}",beta);

    let s_prime = beta * patch_stride;

    let gamma = ceiling_div(array_size,patch_stride);
    println!("total # patches: {}",gamma);

    let w = (gamma - 1) * patch_stride + patch_size;
    println!("padded array length: {}",w);

    let mut sum = 0;
    for i in 0..beta {
        let gamma_prime = ceiling_div(array_size - i*patch_stride,s_prime);
        // gamma is the number of patches of each sub-problem
        println!("partition {} # of patches: {}",i,gamma_prime);
        sum += gamma_prime;
    }

    assert_eq!(sum,gamma);

    let partition_idx = 1;
    let patch_idx = 1;
    let partition_start = partition_idx*patch_stride;
    
    let start_address = partition_start + patch_idx * beta;
    let range = start_address..(start_address + patch_size);

    println!("start address: {}",start_address);
    println!("range: {:?}",range);


}