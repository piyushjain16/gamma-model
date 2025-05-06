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
import Myloss
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as SSIM
from math import log10, sqrt
import glob
import logging
from network_swinir import SwinIR as net
# from network_ffdnet import FFDNet as net


logging.basicConfig(filename="gamma_swinIR.log", filemode="a", format="%(levelname)s: %(message)s",level=logging.INFO)
logger=logging.getLogger()

torch.autograd.set_detect_anomaly(True)
 
# def PSNR(original, compressed): 
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
    
# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()

def lowlight(image_path, label_path,gamma_net,swainIR_model):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)
    label_lowlight = Image.open(label_path)

    # Normalize and prepare tensors for DCE network
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1).unsqueeze(0).cuda()

    label_lowlight = (np.asarray(label_lowlight) / 255.0)
    label_lowlight = torch.from_numpy(label_lowlight).float().permute(2, 0, 1).unsqueeze(0).cuda()
  
    enhanced_image, _ , _ = gamma_net(data_lowlight)

    # with torch.no_grad():
    print("enhanced_image shape",enhanced_image.shape," img_lowlight shape",data_lowlight.shape," label_lowlight shape",label_lowlight.shape)
    denoised_image = swainIR_model(enhanced_image)
    print("denoise_image shape:", denoised_image.shape)
    
    # Remove batch dimension and convert to numpy (H, W, C) format
    enhanced_imagenp = denoised_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    label_lowlightnp = label_lowlight.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    del enhanced_image,denoised_image,label_lowlight

    # Ensure the image is in range [0, 255] as uint8
    enhanced_imagenp = (enhanced_imagenp * 255).astype(np.uint8)
    label_lowlightnp = (label_lowlightnp * 255).astype(np.uint8)
    
    # Calculate PSNR and SSIM
    psnr = PSNR(enhanced_imagenp, label_lowlightnp)
    # ssim = SSIM(enhanced_imagenp, label_lowlightnp, win_size=3, multichannel=True)
    ssim = calculate_ssim(enhanced_imagenp, label_lowlightnp)
    # print("PSNR :", psnr, " SSIM :", ssim)
    return psnr, ssim

def test_lowlightimage(gamma_net,config,epoch,swainIR_model):
    with torch.no_grad():
        sum_psnr = 0
        sum_ssim = 0
        test_size = 0
        filePath = config.test_image_path
        file_name = 'low'
        label_name = 'high'
        test_list = glob.glob(filePath + file_name + "/*")

        for image in test_list:
            image_label = image.replace(file_name, label_name)
            psnr, ssim = lowlight(image, image_label,gamma_net,swainIR_model)
            test_size += 1
            sum_psnr += psnr
            sum_ssim += ssim
        logger.info("Epoch :"+ str(epoch) +", PSNR :" + str(sum_psnr / test_size) + ", SSIM :" + str(sum_ssim / test_size))
        print("avg PSNR :", sum_psnr / test_size, " avg SSIM:", sum_ssim / test_size)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

    os.environ['CUDA_VISIBLE_DEVICES']='0'

    gamma_net = gamma_model.enhance_net_nopool().cuda()

    gamma_net.apply(weights_init)
    if config.load_pretrain == True:
        gamma_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    
    # ffd_model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    # ffd_model.load_state_dict(torch.load(model_path), strict=True)
    # ffd_model.eval()
    # for k, v in ffd_model.named_parameters():
    #     v.requires_grad = False
    # ffd_model = ffd_model.cuda()
    # border = 0
    # window_size = 8
    swainIR_model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key_g = 'params_ema'
    pretrained_model = torch.load(config.pretrain_swainir_dir)
    swainIR_model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    # swainIR_model.load_state_dict(torch.load(config.pretrain_swainir_dir), strict=True)
    swainIR_model.eval()
    swainIR_model = swainIR_model.cuda()

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()

    L_exp = Myloss.L_exp(16,0.6)
    L_TV = Myloss.L_TV()


    optimizer = torch.optim.Adam(gamma_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    gamma_net.train()

    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cuda()

            enhanced_image,gamma ,beta  = gamma_net(img_lowlight)  # enhanced_image shape torch.Size([8, 3, 256, 256])  img_lowlight shape torch.Size([8, 3, 256, 256])

            # print("enhanced_image shape:", enhanced_image.shape)

            # sigma = torch.full((enhanced_image.shape[0],1,1,1), 15/255.).type_as(enhanced_image)

            # # with torch.no_grad():
            # denoised_image = ffd_model(enhanced_image, sigma) # img_L shape: torch.Size([1, 3, 400, 600])  sigma shape:  torch.Size([1, 1, 1, 1])

            # print("denoise_image shape:", denoised_image.shape)
            
            Loss_TV = 1000*(L_TV(gamma) + L_TV(beta))
            
            loss_spa = 5*torch.mean(L_spa(enhanced_image, img_lowlight))

            loss_col = 10*torch.mean(L_color(enhanced_image))

            loss_exp = 10*torch.mean(L_exp(enhanced_image))
            # best_loss
            loss =  Loss_TV + loss_spa + loss_col + loss_exp
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(gamma_net.parameters(),config.grad_clip_norm)
            optimizer.step()

            print("started testing")
            test_lowlightimage(gamma_net,config,epoch,swainIR_model)

            if ((iteration+1) % config.display_iter) == 0:
                logger.info("Loss at iteration " + str(iteration+1) + " : " + str(loss.item()) + " Loss_TV :" + str(Loss_TV.item()) + " loss_spa :" + str(loss_spa.item()) + " loss_col :" + str(loss_col.item()) + " loss_exp :" + str(loss_exp.item()))
                logger.info("Epoch :"+ str(epoch) +", Loss at iteration" + str(iteration+1) + ":" + str(loss.item()))
                # print("Loss at iteration " + str(iteration+1) + " : " + str(loss.item()) + " Loss_TV :" + str(Loss_TV.item()) + " loss_spa :" + str(loss_spa.item()) + " loss_col :" + str(loss_col.item()) + " loss_exp :" + str(loss_exp.item()))
                print("Loss at iteration ", iteration+1, " : ", loss.item())
            if ((iteration+1) % config.snapshot_iter) == 0:
                torch.save(gamma_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
        test_lowlightimage(gamma_net,config,epoch,swainIR_model) 		




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default=r"E:/python/Zero-DCE-master/Zero-DCE_code/data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="E:/python/Zero-DCE-master/Zero-DCE_code/snapshot_gamma_with_swinir/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--test_image_path', type=str, default= "E:/python/Zero-DCE-master/Zero-DCE_code/data/Lol_v1_test/")
    parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")
    parser.add_argument('--pretrain_swainir_dir', type=str, default= r"E:\python\Zero-DCE-master\Zero-DCE_code\swainIR_weights\003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)