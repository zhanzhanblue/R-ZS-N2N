import os
from utils.dnd_denoise import denoise_srgb
from utils.pytorch_wrapper import pytorch_denoiser
from utils.bundle_submissions import bundle_submissions_srgb

if __name__=='__main__':
    datafold = './datasets/DND/dnd_2017'
    
    device = 'cuda:0'
    roll_d = 12
    save_name = 'test'
    if not os.path.exists('./results/DND/{}'.format(save_name)):
        os.mkdir('./results/DND/{}'.format(save_name))
        
    outfold = './results/DND/Submit'
    if not os.path.exists(outfold):
        os.mkdir(outfold)
    
    idx = 0
    denoiser = pytorch_denoiser(device)
    denoise_srgb(denoiser, datafold, outfold, roll_d=roll_d, save_name=save_name)
    bundle_submissions_srgb(outfold)