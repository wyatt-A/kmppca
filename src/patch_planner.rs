use crate::ceiling_div;

struct PatchPlanner {
    patch_size:[usize;3],
    patch_stride:[usize;3],
    volume_size:[usize;3],
}


impl PatchPlanner {
    fn n_patches(&self) -> [usize;3] {
        [
            ceiling_div(self.volume_size[0], self.patch_stride[0]),
            ceiling_div(self.volume_size[1], self.patch_stride[1]),
            ceiling_div(self.volume_size[2], self.patch_stride[2])
        ]
    }

    fn patch_size(&self) -> usize {
        self.patch_size.iter().product()
    }

    fn n_total_patches(&self) -> usize {
        self.n_patches().into_iter().product()
    }

    // fn n_patches_per_partition(&self) -> usize {
    //     self.n_total_patches() / self.n_total_partitions()
    // }

    // a partition contains no overlapping patches
    fn n_partitions(&self) -> [usize;3] {
        [
            self.patch_size[0] - self.patch_stride[0] + 1,
            self.patch_size[1] - self.patch_stride[1] + 1,
            self.patch_size[2] - self.patch_stride[2] + 1,
        ]
    }

    // fn patches_per_partition(&self) -> [usize;3] {
    //     let n_patches = self.n_patches();
    //     let n_partitions = self.n_partitions();
    //     [
    //         n_patches[0] / n_partitions[0],
    //         n_patches[1] / n_partitions[1],
    //         n_patches[2] / n_partitions[2],
    //     ]
    // }

    fn n_total_partitions(&self) -> usize {
        self.n_partitions().into_iter().product()
    }



    fn partition_stride(&self) -> [usize;3] {
        [
            2*self.patch_size[0] - self.patch_stride[0],
            2*self.patch_size[1] - self.patch_stride[1],
            2*self.patch_size[2] - self.patch_stride[2],
        ]
    }


    fn global_patch_subscr(&self,partition_subscr:[usize;3],local_patch_subscr:[usize;3]) -> [usize;3] {
        let part_stride = self.n_partitions();
        let partition_size = self.partition_size(partition_subscr);

        assert!(
            local_patch_subscr.into_iter().zip(partition_size).all(|(x,y)| x < y),
            "local patch subscript {:?} out of bounds",local_patch_subscr
        );

        assert!(
            partition_subscr.into_iter().zip(part_stride).all(|(x,y)| x < y),
            "partition subscript {:?} out of bounds",partition_subscr
        );

        [
            part_stride[0] * local_patch_subscr[0] + partition_subscr[0],
            part_stride[1] * local_patch_subscr[1] + partition_subscr[1],
            part_stride[2] * local_patch_subscr[2] + partition_subscr[2],
        ]
    }

    fn vol_indices(&self,partition_subscr:[usize;3],local_patch_subscr:[usize;3],indices:&mut [usize]) {
        self.grid_subscript_to_grid_indices(
            self.global_patch_subscript_to_grid_subscript(
                self.global_patch_subscr(partition_subscr,local_patch_subscr)
            )
            , indices
        );
    }

    /// size of the partition in units of patches
    fn partition_size(&self,partition_subscr:[usize;3]) -> [usize;3] {
        let n_partitions = self.n_partitions();
        let partition_stride = self.partition_stride();
        [
            ceiling_div(self.volume_size[0] - partition_subscr[0] * n_partitions[0], partition_stride[0]),
            ceiling_div(self.volume_size[1] - partition_subscr[1] * n_partitions[1], partition_stride[1]),
            ceiling_div(self.volume_size[2] - partition_subscr[2] * n_partitions[2], partition_stride[2]),
        ]
    }


    fn global_patch_subscript_to_grid_subscript(&self,patch_subscr:[usize;3]) -> [usize;3] {
        [
            patch_subscr[0] * self.patch_stride[0],
            patch_subscr[1] * self.patch_stride[1],
            patch_subscr[2] * self.patch_stride[2],
        ]
    }

    /// indices must have the same length as the number of voxels in a patch
    fn grid_subscript_to_grid_indices(&self,grid_subscr:[usize;3],indices:&mut [usize]) {
        let padded_vol_size = self.padded_volume_size();
        let mut i = 0;
        for ix in grid_subscr[0]..(grid_subscr[0]+self.patch_size[0]) {
            for iy in grid_subscr[1]..(grid_subscr[1]+self.patch_size[1]) {
                for iz in grid_subscr[2]..(grid_subscr[2]+self.patch_size[2]) {
                    indices[i] = array_utils::sub_to_idx_col_major(
                        &[ix,iy,iz],
                        &padded_vol_size // this has to be the padded volume size
                    ).unwrap();
                    i += 1;
                }
            }
        }
    }

    fn padded_volume_size(&self) -> [usize;3] {
        let n_patches = self.n_patches();
        [
            (n_patches[0] - 1) * self.patch_stride[0] + self.patch_size[0],
            (n_patches[1] - 1) * self.patch_stride[1] + self.patch_size[1],
            (n_patches[2] - 1) * self.patch_stride[2] + self.patch_size[2],
        ]
    }


}


#[test]
fn test() {

    let p = PatchPlanner {
        patch_size: [10,10,10],
        patch_stride: [10,10,10],
        volume_size: [197,120,120],
    };

    println!("{:?}",p.n_partitions());
    println!("{:?}",p.n_patches());
    println!("{:?}",p.padded_volume_size());

    let mut indices = vec![0usize;p.patch_size()];
    p.vol_indices([0,0,0], [0,0,0], &mut indices);
    println!("{:?}",indices);

}