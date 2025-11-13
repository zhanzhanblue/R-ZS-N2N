# import wget
import os
import scipy.io
import numpy as np
import pandas as pd
import base64
from sidd_validation import denoising
from PIL import Image

def my_srgb_denoiser(x, roll_d=12):
    """TODO: Implement your own sRGB denoiser here."""
    
    x = x.astype(np.float32) / 255.0
    tmp_clean = np.zeros_like(x)
    
    device = 'cuda:0'
    show = False
    R=False
    E=True
    *_, denoised_img = denoising(x, tmp_clean, padding=12, pd_tr=5, roll_d=roll_d, pd_te=2, E=E, R=R, device=device, show=show)
    
    x = np.round(denoised_img*255).astype(np.uint8)
    
    return x.copy()

def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

def base64string_to_array(base64string, array_dtype, array_shape):
    decoded_bytes = base64.b64decode(base64string)
    decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
    decoded_array = decoded_array.reshape(array_shape)
    return decoded_array


if __name__=='__main__':
    # Download input file, if needed.
    url = 'https://competitions.codalab.org/my/datasets/download/0d8a1e68-155d-4301-a8cd-9b829030d719'
    input_file = './datasets/SIDD_Benchmark/BenchmarkNoisyBlocksSrgb.mat'
    if os.path.exists(input_file):
        print(f'{input_file} exists. No need to download it.')
    else:
        print("Error ! No dataset")
        print('Downloading input file BenchmarkNoisyBlocksSrgb.mat...')
        # wget.download(url, input_file)
        print('Downloaded successfully.')
    
    # Read inputs.
    key = 'BenchmarkNoisyBlocksSrgb'
    inputs = scipy.io.loadmat(input_file)
    inputs = inputs[key]
    print(f'inputs.shape = {inputs.shape}')
    print(f'inputs.dtype = {inputs.dtype}')
    
    save_name = 'test'
    roll_d = 12
    idx = 0
    if not os.path.exists('./results/SIDD_Benchmark/{}'.format(save_name)):
        os.mkdir('./results/SIDD_Benchmark/{}'.format(save_name))
    
    # Denoising.
    output_blocks_base64string = []
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            
            in_block = inputs[i, j, :, :, :]
            out_block = my_srgb_denoiser(in_block, roll_d=roll_d)
            Image.fromarray(out_block, mode='RGB').save('./results/SIDD_Benchmark/{}/{:0>4}.png'.format(save_name, idx))
            idx += 1
            
            assert in_block.shape == out_block.shape
            assert in_block.dtype == out_block.dtype
            out_block_base64string = array_to_base64string(out_block)
            output_blocks_base64string.append(out_block_base64string)

    # Save outputs to .csv file.
    output_file = './results/SIDD_Benchmark/Submit/{}.csv'.format(save_name)
    print(f'Saving outputs to {output_file}')
    output_df = pd.DataFrame()
    n_blocks = len(output_blocks_base64string)
    print(f'Number of blocks = {n_blocks}')
    output_df['ID'] = np.arange(n_blocks)
    output_df['BLOCK'] = output_blocks_base64string

    output_df.to_csv(output_file, index=False)

    # TODO: Submit the output file SubmitSrgb.csv at 
    # kaggle.com/competitions/sidd-benchmark-srgb-psnr
    print('TODO: Submit the output file SubmitSrgb.csv at')
    print('kaggle.com/competitions/sidd-benchmark-srgb-psnr')

    print('Done.')