import os
import numpy as np
import matplotlib.pyplot as plt
import glob

def print_results(img_path, mask_path, result_path, ext):
    """plot all image-mask-result triple load from .npy files

    Args:
        img_path (str): path to the image folder
        mask_path (str): path to the mask folder
        result_path (str): path to the result folder, which will be looped over
    """
    num_img = 10
    _, axs = plt.subplots(num_img, 3, sharex=True, sharey=True , figsize = (10,4*num_img))
    
    for i, img_file in enumerate(glob.glob(result_path+f'/*.{ext}')):

        img = np.load(os.path.join(img_path, os.path.basename(img_file))).transpose(1,2,0)
        mask = np.load(os.path.join(mask_path, os.path.basename(img_file)))
        result = np.load(os.path.join(result_path, os.path.basename(img_file))).transpose(1,2,0)
        if i == num_img:
            break
        axs[i,0].imshow(img)
        axs[i,1].imshow(mask)
        axs[i,2].imshow(result)
    plt.tight_layout()
    plt.show()
    
    
if __name__=='__main__':
    img_path = '/home/anhho/data/TrGiang_Workspace/Diffusion_models/syn10-diffusion/experiments/inpainting/demo_requests/parcel/reso_05/results/images_tiles/740_beta2_0/'
    mask_path = '/home/anhho/data/TrGiang_Workspace/Diffusion_models/syn10-diffusion/experiments/inpainting/demo_requests/parcel/reso_05/results/masks_tiles/740_beta2_0'
    res_path = '/home/anhho/data/TrGiang_Workspace/Diffusion_models/syn10-diffusion/experiments/inpainting/demo_requests/parcel/reso_05/results/results/740_beta2_0/'

    print_results(img_path, mask_path, res_path, 'npy')