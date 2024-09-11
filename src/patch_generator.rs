#[cfg(test)]
mod tests {

    use crate::patch_generator::PatchGenerator;
    use cfl::ndarray::{ArrayD,ShapeBuilder};
    use cfl::num_complex::Complex32;
    use rand::Rng;

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
}

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



impl PatchGenerator {

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

    pub fn patch_size(&self) -> usize {
        self.x.window_len * self.y.window_len * self.z.window_len
    }

    pub fn n_patches(&self) -> usize {
        self.x.n_iter() * self.y.n_iter() * self.z.n_iter()
    }

    fn iter(&self) -> PatchIterator {
        self.into_iter()
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



#[derive(Debug,Clone)]
pub struct CircularWindow1D {
    array_len:usize,
    window_len: usize,
    stride: usize,
}

impl CircularWindow1D {

    pub fn new(array_len:usize,window_len:usize,stride:usize) -> Self {
        Self {
            array_len,
            window_len,
            stride,
        }
    }
    pub fn nth(&self,stride_idx:usize) -> Option<Vec<usize>> {
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

    pub fn n_iter(&self) -> usize {
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
pub struct CircularWindowIterator<'a> {
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

fn ceiling_div(a:usize,b:usize) -> usize {
    (a + b - 1) / b
}