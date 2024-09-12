pub mod patch_generator;
use std::{cell::RefCell, collections::HashMap, fs::File, io::{Read, Seek}, mem::size_of, path::Path};

use cfl::num_complex::Complex32;
use ndarray_linalg::{c32, SVD};

#[cfg(test)]
mod tests {

    use cfl::{ndarray::{Array3, ArrayD, Axis, Ix2, Ix3, ShapeBuilder}, CflReader};
    use ndarray_linalg::{c32, SVD};
    use cfl::num_complex::Complex32;
    use crate::{patch_generator::PatchGenerator, VolumeReader};

    #[test]
    fn it_works() {
        let x = cfl::to_array("/Users/Wyatt/scratch/4D_test_data/raw/i00", true).unwrap().into_dimensionality::<Ix3>().unwrap();
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

        let mut data_set = vec![];
        for i in 0..67 {
            let filename = format!("/home/wyatt/nordic_test/i{:02}",i);
            data_set.push(
                CflReader::new(filename).unwrap()
            )
        }

        let patch_gen = PatchGenerator::new(dims, [11,11,11], [8,8,8]);

        let patch_batch_size = 200;

        let mut patch_d = Array3::from_elem((patch_gen.patch_size(),data_set.len(),patch_batch_size).f(), Complex32::ZERO);

        let mut it = patch_gen.into_iter();

        patch_d.axis_iter_mut(Axis(2)).for_each(|mut patch|{
            let patch_idx = it.next().unwrap();
            for (mut col,vol) in patch.axis_iter_mut(Axis(1)).zip(data_set.iter_mut()) {
                vol.read_into(&patch_idx, col.as_slice_memory_order_mut().unwrap()).unwrap()
            }
        });

        println!("patch data: {:?}",patch_d);

    }
    

}


#[derive(Debug)]
struct VolumeReader {
    file: File,
    data_cache:HashMap<usize,Complex32>,
    num_elements:usize,
    read_buffer:[u8;8],
    max_cache_size:Option<usize>,
    cache_hits:usize,
    cache_misses:usize,
}

impl VolumeReader {
    pub fn new(cfl:impl AsRef<Path>) -> Self {

        let dims = cfl::get_dims(cfl.as_ref().with_extension("hdr")).unwrap();

        let data = cfl::to_array(&cfl, true).unwrap();
        
        // let data_cache:HashMap<usize,Complex32> = HashMap::from_iter(
        //     data.as_slice_memory_order().unwrap().iter().enumerate().map(|(idx,val)|(idx,*val))
        // );
        
        Self {
            file: File::open(cfl.as_ref().with_extension("cfl")).unwrap(),
            data_cache: HashMap::new(),
            read_buffer: [0u8;8],
            max_cache_size: None,
            cache_hits:0,
            cache_misses:0,
            num_elements: dims.iter().product()
        }       
    }

    pub fn fill(&mut self,indices:&[usize],dst:&mut [Complex32]) {
        indices.iter().zip(dst.iter_mut()).for_each(|(idx,val)|{
            *val = self.get(*idx).unwrap()
        });
    }

    pub fn get(&mut self,idx:usize) -> Option<Complex32> {

        if idx >= self.num_elements {
            println!("index out of bounds {} >= {}",idx,self.num_elements);
            return None
        }

        if let Some(entry) = self.data_cache.get(&idx) {
            self.cache_hits += 1;
            return Some(*entry)
        }else {
            self.file.seek(std::io::SeekFrom::Start((idx * size_of::<Complex32>()) as u64)).unwrap();
            self.file.read_exact(&mut self.read_buffer).unwrap();

            let mut val = [Complex32::ZERO];
        
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.read_buffer.as_ptr(),
                    val.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<Complex32>(),
                );
            }

            self.data_cache.insert(idx, val[0]);
            self.cache_value(idx, val[0]);
            self.cache_misses += 1;
            Some(val[0])
        }
    }

    fn cache_value(&mut self, idx: usize, value: Complex32) {
        // Insert into cache
        if let Some(max_size) = self.max_cache_size {
            if self.data_cache.len() >= max_size {
                // Evict a value if the cache is full (e.g., use LRU)
                self.evict();
            }
        }

        self.data_cache.insert(idx, value);
    }

    // A simple eviction policy: remove the first inserted element (you can optimize this later)
    fn evict(&mut self) {
        if let Some((&first_key, _)) = self.data_cache.iter().next() {
            self.data_cache.remove(&first_key);
        }
    }

}