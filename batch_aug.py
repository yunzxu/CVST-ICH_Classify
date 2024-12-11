import numpy as np
import os 
from scipy.ndimage import zoom
import nibabel as nib
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from skimage import data
from batchgenerators.dataloading.data_loader import DataLoaderBase,SlimDataLoaderBase
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter


default_3D_augmentation_params = {
"selected_data_channels": None,
"selected_seg_channels": None,

"do_elastic": True,
"elastic_deform_alpha": (0., 900.),
"elastic_deform_sigma": (9., 13.),
"p_eldef": 0.4,

"do_scaling": True,
"scale_range": (0.85, 1.25),
"independent_scale_factor_for_each_axis": False,
"p_independent_scale_per_axis": 1,
"p_scale": 0.3,

"do_rotation": True,
"rotation_x": (-45. / 360 * 2. * np.pi, 45. / 360 * 2. * np.pi),
"rotation_y": (-45. / 360 * 2. * np.pi, 45. / 360 * 2. * np.pi),
"rotation_z": (-45. / 360 * 2. * np.pi, 45. / 360 * 2. * np.pi),
"rotation_p_per_axis": 1,
"p_rot": 0.4,

"random_crop": False,
"random_crop_dist_to_border": None,

"do_gamma": True,
"gamma_retain_stats": True,
"gamma_range": (0.7, 1.5),
"p_gamma": 0.4,

"do_mirror": True,
"mirror_axes": (0, 1, 2),

"dummy_2D": False,
"mask_was_used_for_normalization": None,
"border_mode_data": "constant",

"all_segmentation_labels": None,  # used for cascade
"move_last_seg_chanel_to_data": False,  # used for cascade
"cascade_do_cascade_augmentations": False,  # used for cascade
"cascade_random_binary_transform_p": 0.4,
"cascade_random_binary_transform_p_per_label": 1,
"cascade_random_binary_transform_size": (1, 8),
"cascade_remove_conn_comp_p": 0.2,
"cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
"cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

"do_additive_brightness": False,
"additive_brightness_p_per_sample": 0.15,
"additive_brightness_p_per_channel": 0.5,
"additive_brightness_mu": 0.0,
"additive_brightness_sigma": 0.1,

"num_threads": 12 if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
"num_cached_per_thread": 1,
}


def transforms_create(Data,patch_size=[96,96,96],batch_size=4,train_patch=True,pro=0):##直接使用transform进行变化，不使用它的生成器
    # print('transforms_create')
    params=default_3D_augmentation_params
    patch_size_spatial=patch_size
    order_seg=3
    border_val_seg=0
    order_data=3
    Spatial=SpatialTransform(
    patch_size_spatial, patch_center_dist_from_border=None,
    do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
    sigma=params.get("elastic_deform_sigma"),
    do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
    angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
    do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
    border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
    border_mode_seg="constant", border_cval_seg=border_val_seg,
    order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
    p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
    independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    )
    GaussianN=GaussianNoiseTransform(p_per_sample=0.1)
    GaussianB=GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                p_per_channel=0.5)
    Brightness=BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)

    Contrast=ContrastAugmentationTransform(p_per_sample=0.15)
    SimulateLow=SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=None)
    
    GammaTrans1 = GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                    p_per_sample=0.1)  # inverted gamma
    
    GammaTrans2 = GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                    p_per_sample=params["p_gamma"])
    if params.get("do_mirror") or params.get("mirror"):
       MirrorT=MirrorTransform(params.get("mirror_axes"))
    #print('transform')
    if np.random.uniform() > pro:
        Data=Spatial(**Data)
    if np.random.uniform() > pro:
        Data=GaussianN(**Data)
        Data=GaussianB(**Data)
    if np.random.uniform() > pro:
        Data=Brightness(**Data)
    if np.random.uniform() > pro:
        Data=Contrast(**Data)
    if np.random.uniform() > pro:
        Data=SimulateLow(**Data)
    if np.random.uniform() > pro:
        Data=GammaTrans1(**Data)
    if np.random.uniform() > pro:
        Data=GammaTrans2(**Data)
    if np.random.uniform() > pro:
        Data=MirrorT(**Data)
    del Spatial,GaussianN,GaussianB,Brightness,Contrast,SimulateLow,GammaTrans1,GammaTrans2,MirrorT
    return Data



# if __name__ == '__main__':

#         X_patch = X_patch[np.newaxis,np.newaxis,:,:,:]

#         Y_patch = Y_patch[np.newaxis,np.newaxis,:,:,:]
#         data={'data':np.array(X_patch),'seg':np.array(Y_patch)}

#         if self.aug:
#             if np.random.uniform() > 0.5:
#             # print('transform')
#                 data=transforms_create(data,patch_size=[96,96,96],batch_size=1,train_patch=True)
        
#         X_patch=data['data'][0]
#         Y_patch=data['seg'][0]