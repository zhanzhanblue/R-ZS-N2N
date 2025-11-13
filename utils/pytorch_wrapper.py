 # Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.

import numpy as np
from sidd_validation import denoising

def pytorch_denoiser(device,):
    def wrap_denoiser(Inoisy, roll_d, nlf): 
                
        tmp_clean = np.zeros_like(Inoisy)
        *_, denoised_img = denoising(Inoisy, tmp_clean, padding=14, pd_tr=5, roll_d=roll_d, pd_te=2, R=False, E=False, device=device, show=False)
  
        return denoised_img

    return wrap_denoiser
