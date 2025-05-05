import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import gamma_model
# import model
# import retinex_gamma
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import cv2
from skimage.metrics import structural_similarity as SSIM
from math import log10, sqrt
from network_ffdnet import FFDNet as net

# def PSNR(original, compressed): 
#     original = np.array(original)
#     compressed = np.array(compressed)
#     mse = np.mean((original - compressed) ** 2) 
#     if mse == 0:  # MSE is zero means no noise is present in the signal
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * log10(max_pixel / sqrt(mse)) 
#     return psnr

def PSNR(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * log10(255.0 / sqrt(mse))

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return SSIM(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(SSIM(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return SSIM(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def lowlight(image_path, label_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)
    label_lowlight = Image.open(label_path)

    # Normalize and prepare tensors for DCE network
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1).unsqueeze(0).cuda()

    label_lowlight = (np.asarray(label_lowlight) / 255.0)
    label_lowlight = torch.from_numpy(label_lowlight).float().permute(2, 0, 1).unsqueeze(0).cuda()

    gamma_net = gamma_model.enhance_net_nopool().cuda()
    gamma_net.load_state_dict(torch.load('E:\python\Zero-DCE-master\Zero-DCE_code\snapshot_gamma_with_gamma_beta_map_ffd\Epoch0.pth', weights_only=True))

    n_channels = 3
    nc = 96
    nb = 12
    model_path = r"E:\python\Zero-DCE-master\Zero-DCE_code\FFD_net_code\ffdnet_color.pth"
    
    ffd_model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    ffd_model.load_state_dict(torch.load(model_path), strict=True)
    ffd_model = ffd_model.cuda()

    enhanced_image,_ ,_= gamma_net(data_lowlight)

    sigma = torch.full((enhanced_image.shape[0],1,1,1), 15/255.).type_as(enhanced_image)

    denoised_image = ffd_model(enhanced_image, sigma)

    # retinex_gamma_model = retinex_gamma.FullModel().cuda()
    # retinex_gamma_model.load_state_dict(torch.load('E:\python\Zero-DCE-master\Zero-DCE_code\snapshots_retinex_gamma\Epoch2.pth', weights_only=True))
  
    # R, I, out, gamma,enhanced_image = retinex_gamma_model(data_lowlight)
    
    # Remove batch dimension and convert to numpy (H, W, C) format
    enhanced_imagenp = denoised_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    label_lowlightnp = label_lowlight.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    # R = R.squeeze(0).permute(1,2, 0).cpu()
    # I = I.squeeze(0).permute(1, 2, 0).cpu()
    # out = out.squeeze(0).permute(1, 2, 0).cpu()

    # Ensure the image is in range [0, 255] as uint8
    enhanced_imagenp = (enhanced_imagenp * 255).astype(np.uint8)
    label_lowlightnp = (label_lowlightnp * 255).astype(np.uint8)
    
    # Calculate PSNR and SSIM
    psnr = PSNR(enhanced_imagenp, label_lowlightnp)
    # ssim = SSIM(enhanced_imagenp, label_lowlightnp, win_size=3, multichannel=True)
    ssim = calculate_ssim(enhanced_imagenp, label_lowlightnp)
    print("PSNR :", psnr, " SSIM :", ssim)
    # result_path = image_path.replace('low', 'result_retinex_gamma')
    # result_path_R = image_path.replace('low', 'result_r')
    # result_path_I = image_path.replace('low', 'result_i')
    # result_path_out = image_path.replace('low', 'result_out')
    result_path = image_path.replace('low', 'result_gamma_with_gamma_beta_map_ffd_epoch0',1)


    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(result_path)):
      os.makedirs(os.path.dirname(result_path))
    # if not os.path.exists(os.path.dirname(result_path_R)):
    #     os.makedirs(os.path.dirname(result_path_R))
    # if not os.path.exists(os.path.dirname(result_path_I)):
    #     os.makedirs(os.path.dirname(result_path_I))
    # if not os.path.exists(os.path.dirname(result_path_out)):
    #     os.makedirs(os.path.dirname(result_path_out))
    # Save enhanced image using torchvision
    enhanced_imagenp = enhanced_imagenp.astype(np.float32) / 255.0  # Normalize to [0, 1]
    torchvision.utils.save_image(torch.from_numpy(enhanced_imagenp).permute(2, 0, 1), result_path)
    # torchvision.utils.save_image(R.permute(2, 0, 1), result_path_R)
    # torchvision.utils.save_image(I.permute(2, 0, 1), result_path_I)
    # torchvision.utils.save_image(out.permute(2, 0, 1), result_path_out)

    return psnr, ssim

if __name__ == '__main__':
    with torch.no_grad():
        sum_psnr = 0
        sum_ssim = 0
        test_size = 0
        filePath = 'E:/python/Zero-DCE-master/Zero-DCE_code/data/Lol_v1_test/'
        file_name = 'low'
        label_name = 'high'
        test_list = glob.glob(filePath + file_name + "/*")
        dataset_name = 'Lol_v1_test'

        for image in test_list:
            image_label = "" 
            # for lol v2 
            if(dataset_name == 'Lol_v2_test'):
                # Get just the filename (e.g., low00690.png)
                filename = os.path.basename(image)
                
                # Construct new label path using label_name and filename
                image_label = os.path.join(filePath, label_name, filename.replace('low', 'normal', 1))
            elif (dataset_name == 'Lol_v1_test') :
                # for lol v1 
                image_label = image.replace(file_name, label_name)
            
            psnr, ssim = lowlight(image, image_label)
            test_size += 1
            sum_psnr += psnr
            sum_ssim += ssim
        
        print("avg PSNR :", sum_psnr / test_size, " avg SSIM:", sum_ssim / test_size)

 
# def lowlight(image_path):
# 	os.environ['CUDA_VISIBLE_DEVICES']='0'
# 	data_lowlight = Image.open(image_path)

 

# 	data_lowlight = (np.asarray(data_lowlight)/255.0)


# 	data_lowlight = torch.from_numpy(data_lowlight).float()
# 	data_lowlight = data_lowlight.permute(2,0,1)
# 	data_lowlight = data_lowlight.cuda().unsqueeze(0)

# 	DCE_net = model.enhance_net_nopool().cuda()
# 	DCE_net.load_state_dict(torch.load('Zero-DCE-master\Zero-DCE_code\snapshots\Epoch99.pth'))
# 	start = time.time()
# 	_,enhanced_image,_ = DCE_net(data_lowlight)

# 	end_time = (time.time() - start)
# 	print(end_time)
# 	image_path = image_path.replace('test_data','result')
# 	result_path = image_path
# 	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
# 		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

# 	torchvision.utils.save_image(enhanced_image, result_path)

# if __name__ == '__main__':
# # test_images
# 	with torch.no_grad():
# 		filePath = 'E:/python/Zero-DCE-master/Zero-DCE_code/data/test_data/'
    
# 		file_list = os.listdir(filePath)

# 		for file_name in file_list:
# 			test_list = glob.glob(filePath+file_name+"/*") 
# 			for image in test_list:
# 				# image = image
# 				print(image)
# 				lowlight(image)

