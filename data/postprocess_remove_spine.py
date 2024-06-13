import os
import nibabel as nib
import numpy as np
from scipy import ndimage
import cc3d
from joblib import Parallel, delayed

ori_label_root_dir = '/data/datasets/nnunet_cmf/Dataset005_cmf_nii/labelsTr'
sat_label_root_dir = '/data/datasets/nnunet_cmf/Dataset005_cmf_nii/SAT_pred/CMF/'

output_dir = '/data/datasets/nnunet_cmf/Dataset005_cmf_nii/labelsTr_masked/'

str_replace = ['carotid_artery', 'vertebrae', 'internal_carotid_artery', 'cervical_vertebrae']


# find all subdirectories in sat_label_root_dir
subdirs = [x[0] for x in os.walk(sat_label_root_dir)]
subdirs = subdirs[1:]

# iterate each subdirectory
def task(subdir):
    print(f"Processing {subdir}")
    
    # get the subject name
    subject_name = os.path.basename(subdir).split('_')[1]
    
    # read in the original label
    ori_label = os.path.join(ori_label_root_dir, subject_name+'.nii.gz')
    nib_ori_label = nib.load(ori_label)
    ori_label_np = nib_ori_label.get_fdata()
    
    # read in each file in the subdirectory
    mask = np.zeros(ori_label_np.shape)
    for s in str_replace:
        sat_label = os.path.join(subdir, s+'.nii.gz')   
        mask += nib.load(sat_label).get_fdata()
        
    # dilate the mask
    mask = mask > 0
    mask = np.array(mask, dtype=np.uint8)
    mask = ndimage.binary_dilation(mask, iterations=1)
        
    # inverse the mask
    mask = mask == 0
    
    # apply the mask to the original label
    ori_label_np = (ori_label_np * mask).astype(np.uint8)
    
    # remove small connected components
    ori_label_np = cc3d.dust(
                ori_label_np, threshold=100, 
                connectivity=26, in_place=False
            )
    
    # only keep the largest K CC
    # ori_label_np, N = cc3d.largest_k(
    #     ori_label_np, k=1, 
    #     connectivity=26, delta=0,
    #     return_N=True,
    # )

    
    # save the new label
    new_label = nib.Nifti1Image(ori_label_np, nib_ori_label.affine, nib_ori_label.header)
    nib.save(new_label, os.path.join(output_dir, subject_name+'_masked.nii.gz'))
    
results = Parallel(n_jobs=12)(delayed(task)(subj) for subj in subdirs)