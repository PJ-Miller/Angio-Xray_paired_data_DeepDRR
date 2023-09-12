# imports
import os, sys
import numpy as np
from pathlib import Path
import nibabel as nib
import nrrd
from PIL import Image
import argparse
import torch

from deepdrr import geo, Volume, MobileCArm, utils
from deepdrr.projector import Projector
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Deep DRR Example')

    parser.add_argument('--volume_path', type=str, default='path to CTA scan',
                        help='path to CT volume')
    
    parser.add_argument('--artery_path', type=str, default='path to CTA scan segmentation map',
                        help='path to artery mask volume')

    parser.add_argument('--drr_path', type=str, default='',
                        help='path to write DRR images')

    parser.add_argument('--vessel_bool', type=int, default=1,
                        help='1 adds artery, 0 not')

    parser.add_argument('--alpha', type=int, default=90,
                        help='C-arm hyperparameter alpha')

    parser.add_argument('--beta', type=int, default=90, 
                        help='C-arm hyperparameter beta')
                
    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    # Polyaxon
    experiment = Experiment()

    data_dir = os.path.join(list(get_data_paths().values())[0])
    data_paths = get_data_paths()
    outputs_path = get_outputs_path()

    volume_path = data_paths['data1'] + args.volume_path
    artery_path = data_paths['data1'] + args.artery_path
    com_output_path = outputs_path + '/sampleDRR_1.png' 

    print(volume_path, artery_path, com_output_path)

    try:

        vol_with = Volume.from_nrrd(Path(volume_path), use_thresholding=True)
        vol_without = Volume.from_nrrd(Path(volume_path), use_thresholding=True)
        artery_mask, artery_mask_header = nrrd.read(Path(artery_path))

    except:
        vol_with = Volume.from_nifti(Path(volume_path), use_thresholding=True)
        vol_without = Volume.from_nifti(Path(volume_path), use_thresholding=True)
        artery_mask = nib.load(Path(artery_path)).get_fdata()



    # mean_softTissue = volume.materials["soft tissue"].mean()
    # mean_Bone = volume.materials["bone"].mean()
    
    if args.vessel_bool == 1:
        vol_with.materials["titanium"] = (1 == artery_mask).astype(np.float32)
        vol_with.materials["bone"] = vol_with.materials["bone"] * (1.0 - vol_with.materials["titanium"])
        vol_with.materials["soft tissue"] = vol_with.materials["soft tissue"] * (1.0 - vol_with.materials["titanium"])
        vol_with.materials["bone"] = vol_with.materials["bone"] * param
        vol_with.materials["soft tissue"] = vol_with.materials["soft tissue"] * 1.0
        vol_with.materials["titanium"] = vol_with.materials["titanium"] * 1.0



        vol_without.materials["titanium"] = (1 == artery_mask).astype(np.float32)
        vol_without.materials["bone"] = vol_without.materials["bone"] * (1.0 - vol_without.materials["titanium"])
        vol_without.materials["bone"] = vol_without.materials["bone"] * param
        vol_without.materials["soft tissue"] = vol_without.materials["soft tissue"] + vol_without.materials["titanium"]
        vol_without.materials["titanium"] = vol_without.materials["titanium"] * 0.0



    carm = MobileCArm(pixel_size=0.7, sensor_width=1440, sensor_height=1440,
                  source_to_isocenter_vertical_distance=742.5,
                  source_to_detector_distance=1259.65,
                  max_alpha=180,
                  min_alpha=-180)
    
    carm.reposition(vol_with.center_in_world)


    for i in range(90):

        carm.move_to(isocenter= [ 0, 0, 0], alpha=float(i*0.5), beta=90, degrees=True)

        
        with Projector(vol_with, carm=carm, neglog=True, spectrum="90KV_AL40") as projector:
            projection = projector()

        with Projector(vol_without, carm=carm, neglog=True, spectrum="90KV_AL40") as projector:
            projection_2 = projector()

        # image_gray1 = 1 - utils.neglog(projection)
        image_gray1 = projection
        # image_gray2 = 1 - utils.neglog(projection_2)
        image_gray2 = projection_2

        image_gray1 = (image_gray1 * 255).astype(np.uint8)
        image_gray2 = (image_gray2 * 255).astype(np.uint8)

        PIL_Image1 = Image.fromarray(image_gray1)
        PIL_Image2 = Image.fromarray(image_gray2)

        PIL_Image1.save(f'{outputs_path}/D1_with_{i}.png')
        PIL_Image2.save(f'{outputs_path}/D1_without_{i}.png')


    print("Fin")

