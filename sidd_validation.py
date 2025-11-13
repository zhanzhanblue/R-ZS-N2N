import os
import time
import torch
import torch.optim as optim
from model.net import network
from utils.utils import *
from utils.get_dataset import get_dataset
from PIL import Image


def loss_func(model, noisy1, noisy2):
    return l1(model(noisy1), noisy2)

def train(model, optimizer, noisy1, noisy2):
    
    loss = loss_func(model, noisy1, noisy2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def denoise(model, noisy_img):
    with torch.no_grad():
        pred = torch.clamp(model(noisy_img),0,1)
        return pred

def run(noisy_img, back_noisy_img, roll_d=12, pd_te=2, R=False, E=True, device='cuda:0'):
    noisy1, noisy2, = get_roll(noisy_img, roll_d=roll_d)
    
    max_epoch=800
    lr=0.015
    
    n_chan = noisy_img.shape[1]
    model = network(n_chan)
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loop = range(max_epoch)
    
    for _ in loop:
        loss_ = train(model, optimizer, noisy1, noisy2)
        if (_+1) % 600 == 0: # 599, next is 600
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.00015
        
    with torch.no_grad():
        back_noisy_img = pixel_shuffle_down_sampling(back_noisy_img, f=pd_te)
        denoised_image = denoise(model, back_noisy_img)
        if R:
            for i in range(1, pd_te*pd_te):
                denoised_image += denoise(model, back_noisy_img.roll(i, dims=(0)))
            denoised_image /= (pd_te*pd_te)
        denoised_image = pixel_shuffle_up_sampling(denoised_image, f=pd_te)
    
    if E:
        return denoise(model, denoised_image)
    return denoised_image

def denoising(noisy_img, clean_img, padding=17, pd_tr=5, roll_d=12, pd_te=2, R=False, E=True, device='cuda:0', show=True):
    
    tic = time.time()
    
    noisy_img = pad_add(noisy_img, pd=padding)
    noisy_img = array2tensor(noisy_img)
    clean_img = array2tensor(clean_img)
    noisy_img = noisy_img.to(device)
    clean_img = clean_img.to(device)
    
    back_noisy_img = noisy_img.clone()
    image_size = clean_img.shape[2]

    noisy_img = pixel_shuffle_down_sampling(noisy_img, f=pd_tr) 
    denoised_img = run(noisy_img, back_noisy_img, roll_d=roll_d, pd_te=pd_te, R=R, E=E, device=device)
    denoised_img = pad_del(denoised_img, pd=(image_size, image_size))
    
    toc = time.time()

    psnr = compute_psnr(denoised_img, clean_img)
    ssim = compute_ssim(denoised_img, clean_img)

    if show: 
        show_result(denoised_img, clean_img, pad_del(back_noisy_img,pd=(image_size, image_size)))
    
    return psnr, ssim, toc-tic, tensor2array(denoised_img)

def proc(dataset_name, save_name, padding=17, pd_tr=5, roll_d=12, pd_te=2, R=False, E=True, save=True, show=True, device='cuda:0'):
    datacount = datasets[dataset_name]
    noisy_images, clean_images = get_dataset(dataset_name)
    
    psnr_sum = 0
    ssim_sum = 0
    idx = 0
    
    if not os.path.exists('./results/SIDD_Valiation/{}'.format(save_name)):
        os.mkdir('./results/SIDD_Valiation/{}'.format(save_name))
    
    with open('./results/SIDD_Valiation/{}.txt'.format(save_name), 'w') as f:
        for i in range(datacount):
            noisy_img, clean_img = noisy_images[i], clean_images[i]
            psnr, ssim, tim, denoised_img = denoising(noisy_img, clean_img, padding=padding, pd_tr=pd_tr, roll_d=roll_d, pd_te=pd_te, R=R, E=E,  device=device, show=show)
            
            if save:
                Image.fromarray(np.clip(denoised_img * 255., 0, 255).astype(np.uint8), mode='RGB').save('./results/SIDD_Valiation/{}/{:0>4}.png'.format(save_name, i))
                idx += 1
            
            print("PICTURE {}: PSNR={:.2f}, ssim={:.3f}, time={:.2f}".format(i, psnr, ssim, tim))
            
            psnr_sum += psnr
            ssim_sum += ssim
            psnr_avg = psnr_sum / (i+1)
            ssim_avg = ssim_sum / (i+1)
            
            print("Average of the PSNR is {:.2f}, SSIM is {:.3f}".format(psnr_avg,ssim_avg))
            f.write("{}: psnr={:.2f}, avg={:.2f}, ssim={:.3f}, avg={:.3f}, time={:.2f}s\n".format(i, psnr, psnr_avg, ssim, ssim_avg, tim))
            f.flush()


if __name__=='__main__':
    
    datasets = {'SIDD_Valiation':1280}
    device = 'cuda:0'
    dataset_name = 'SIDD_Valiation'
    
    padding = 12 # (padding * 2 + w) % pd_tr == 0
    pd_tr = 5 # PD train
    roll_d = 12 # Roll-d
    pd_te = 2 # PD test 
    R=False # R-ZS-N2N (R)
    E=True # R-ZS-N2N (E)
    save=True
    save_name = 'test'
    show = False
    
    if R and E:
        print("Error! do not use Refinement twice!")
        exit(0)
    
    print("process begin!")
    proc(dataset_name, save_name, padding=padding, pd_tr=pd_tr, roll_d=roll_d, pd_te=pd_te, R=R, E=E, save=save, show=show, device=device)