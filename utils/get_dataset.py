import numpy as np
import scipy

def get_dataset(dataset_name):
    '''return image : np.array(W, H, C) '''
    if dataset_name == 'SIDD_Valiation':
        noisy_mat_file_path = './datasets/SIDD_Valiation/ValidationNoisyBlocksSrgb.mat'
        clean_mat_file_path = './datasets/SIDD_Valiation/ValidationGtBlocksSrgb.mat'

        noisy_patches = np.array(scipy.io.loadmat(noisy_mat_file_path, appendmat=False)['ValidationNoisyBlocksSrgb'])
        clean_patches = np.array(scipy.io.loadmat(clean_mat_file_path, appendmat=False)['ValidationGtBlocksSrgb'])
        
        def _load_img_from_np(img):
            return np.ascontiguousarray(img).astype(np.float32)
        
        def load_data(data_idx):
            img_id   = data_idx // 32
            patch_id = data_idx  % 32

            noisy_img = noisy_patches[img_id, patch_id, :].astype(np.float32)
            clean_img = clean_patches[img_id, patch_id, :].astype(np.float32)

            noisy_img = _load_img_from_np(noisy_img)
            clean_img = _load_img_from_np(clean_img)

            return {'real_noisy': noisy_img, 'clean': clean_img }
        
        def get_img_ssid(idx):
            img = load_data(idx)
            noisy_img = img['real_noisy'][::1,:,:] / 255.0
            clean_img = img['clean'] / 255.0
            
            return noisy_img, clean_img 
        
        noisy_imgs = []
        clean_imgs = []
        
        for i in range(1280):
            noisy_imgs.append(get_img_ssid(i)[0])
            clean_imgs.append(get_img_ssid(i)[1])
        
        return noisy_imgs, clean_imgs