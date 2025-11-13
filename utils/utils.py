import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms 
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

def compute_psnr(noisy_img, clean_img):
    noisy_img = noisy_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    clean_img = clean_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return peak_signal_noise_ratio(noisy_img, clean_img, data_range=1)

def compute_ssim(noisy_img, clean_img):
    noisy_img = noisy_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    clean_img = clean_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return structural_similarity(noisy_img, clean_img, channel_axis=2, data_range=1)

def array2tensor(image:np.ndarray):
    return torch.tensor(image).permute(2,0,1).unsqueeze(0)

def tensor2array(image:torch.tensor):
    return np.array(image.detach().squeeze(0).cpu().permute(1,2,0))

def pad_add(image, pd):
    image = cv2.copyMakeBorder(image, pd,pd,pd,pd,cv2.BORDER_REFLECT)
    return image

def pad_del(image:torch.tensor, pd=(256, 256)):
    crop = transforms.CenterCrop(pd)
    return crop(image)

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int):
    '''
    From AP-BSN
    '''
    # single image tensor
    if len(x.shape) == 3:
        exit(0)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        return unshuffled.view(b,c,f,f,w//f,h//f).permute(0,2,3,1,4,5).reshape(b*f*f,c,w//f,h//f)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int):
    '''
    From AP-BSN
    '''
    # single image tensor
    if len(x.shape) == 3:
        exit(0)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b//(f*f),f,f,c,w,h).permute(0,3,1,2,4,5).reshape(b//(f*f),c*f*f,w,h)
        before_shuffle = F.pixel_shuffle(before_shuffle, f)
        return before_shuffle
    
def get_roll(noisy_img:torch.Tensor, roll_d=12):
    b,c,w,h = noisy_img.shape
    f = int(np.sqrt(b))
    n1 = noisy_img.clone()
    n2 = noisy_img.roll(roll_d, dims=0) 
    n1 = n1.reshape(f*f,c,w,h)
    n2 = n2.reshape(f*f,c,w,h)
    return n1, n2

def l1(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.L1Loss()
    return loss(gt,pred)

def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

def show_result(out_denoised_img, clean_img, back_noisy_img):
    '''
    From ZS-N2N.
    '''
    
    denoised = out_denoised_img.detach().cpu().squeeze(0).permute(1,2,0)
    clean = clean_img.detach().cpu().squeeze(0).permute(1,2,0)
    noisy = back_noisy_img.detach().cpu().squeeze(0).permute(1,2,0)

    fig, ax = plt.subplots(1, 3,figsize=(20, 20))

    ax[0].imshow(clean)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Ground Truth')

    ax[1].imshow(noisy)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Noisy Img')
    noisy_psnr = 10*np.log10(1/mse(back_noisy_img, clean_img).item())
    noisy_ssim = compute_ssim(back_noisy_img, clean_img)
    ax[1].set(xlabel= str(round(noisy_psnr,2)) + ' dB ' + str(round(noisy_ssim, 2)))

    ax[2].imshow(denoised)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title('Denoised Img')
    denoisy_psnr = 10*np.log10(1/mse(out_denoised_img,clean_img).item())
    noisy_ssim = compute_ssim(out_denoised_img, clean_img)
    ax[2].set(xlabel= str(round(denoisy_psnr,2)) + ' dB ' + str(round(noisy_ssim, 2)))

    plt.savefig('results/pictures/pic.png')
    plt.show()
