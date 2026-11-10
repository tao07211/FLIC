from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import torch
import torch_radon

def cal_psnr(recon, target):
    datarange = target.max() - target.min()
    return PSNR(target, recon, data_range=datarange)

def cal_ssim(recon, target):
    return SSIM(target, recon, data_range=target.max() - target.min())

def generate_prior_image_for_fanbeam(prior_img, D, device):

    img_size = prior_img.shape[0]
    prior_img_pad = np.zeros((2 * D, 2 * D))
    prior_img_pad[(2 * D - img_size) // 2 : (2 * D + img_size) // 2,
                  (2 * D - img_size) // 2: (2 * D + img_size) // 2] = prior_img
    img_tensor = torch.FloatTensor(prior_img_pad).unsqueeze(0).unsqueeze(0).to(device)
    return img_tensor

def add_noise_to_sino(sino, N0, k):
    sino_ = N0 * np.exp(-sino * k)
    noisy_sino = np.random.poisson(sino_)
    out_sino = -np.log(noisy_sino / N0) / k
    return out_sino

def fan_coordinate(fan_angle, D):
    origin_x = 1
    origin_y = 0
    x = np.linspace(-1, 1, int(2*D)).reshape(-1, 1)                     
    y = np.zeros_like(x)           
    xy_temp = np.concatenate((x, y), axis=-1)           
    xy_temp = np.concatenate((xy_temp, np.ones_like(x)), axis=-1)           
    L = len(fan_angle)
    xy = np.zeros(shape=(L, int(2*D), 2))             
    for i in range(L):
        fan_angle_rad = np.deg2rad(fan_angle[L-i-1])
        M = np.array(
            [
                [np.cos(fan_angle_rad), -np.sin(fan_angle_rad),
                -1*origin_x*np.cos(fan_angle_rad)+origin_y*np.sin(fan_angle_rad)+origin_x],
                [np.sin(fan_angle_rad), np.cos(fan_angle_rad),
                -1*origin_x*np.sin(fan_angle_rad)-origin_y*np.cos(fan_angle_rad)+origin_y],
                [0, 0, 1]
            ]
        )
        temp = xy_temp @ M.T                              
        xy[i, :, :] = temp[:, :2]
    return xy
def build_coordinate_train(xy, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ]
    )
    L, D_2, _ = xy.shape
    xy = xy.reshape(-1, 2)             
    xy = xy @ trans_matrix.T                                   
    xy = xy.reshape(L, D_2, 2)              
    return xy

def parallel_coordinate(det_count, D):

    t = np.linspace(-1, 1, det_count)                        
                          
    s = np.linspace(-1, 1, int(2 * D))                

    xy = np.zeros((det_count, int(2 * D), 2))
    for i in range(det_count):
                                     
        xy[i, :, 0] = t[i]                  
        xy[i, :, 1] = s                   
    return xy                                                       

def build_coordinate_parallel(xy, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array([
        [ np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad), np.cos(angle_rad)]
    ])
    L, D_2, _ = xy.shape
    xy_flat = xy.reshape(-1, 2)
    xy_rot  = xy_flat @ trans_matrix.T
    return xy_rot.reshape(L, D_2, 2)

def fanbeam_projection(img, D, det_count, spacing, num_angles, device):
    import torch.nn.functional as F

    img_size  = img.shape[0]
    N_samples = int(2 * D)                    

    img_padded = np.zeros((N_samples, N_samples), dtype=np.float32)
    pad = (N_samples - img_size) // 2
    img_padded[pad:pad+img_size, pad:pad+img_size] = img
    img_tensor = torch.FloatTensor(img_padded).unsqueeze(0).unsqueeze(0).to(device)

    angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
    xy_base    = parallel_coordinate(det_count, D)                              
    sino       = np.zeros((num_angles, det_count), dtype=np.float32)

    with torch.no_grad():
        for i, angle in enumerate(angles_deg):
            ray    = build_coordinate_parallel(xy_base, angle)
            ray_t  = torch.FloatTensor(ray).unsqueeze(0).to(device)
            coords = ray_t.view(1, 1, -1, 2)
            intens = F.grid_sample(img_tensor, coords, align_corners=True,
                                   mode='bilinear', padding_mode='zeros')
            sino[i] = intens.squeeze().cpu().numpy().reshape(det_count, N_samples).sum(axis=-1)

    return sino                            

def fanbeam_backprojection(sino, D, det_count, spacing, num_angles, device):
    from skimage.transform import iradon
    img_size = int(D / (2 ** 0.5))

    N_samples = int(2 * D)
    sino_norm = sino / N_samples                         
    angles = np.linspace(0, 180, num_angles, endpoint=False)
    fbp = iradon(sino_norm.T, theta=angles, filter_name='ramp', circle=False)
             
    h, w = fbp.shape
    cy, cx = h // 2, w // 2
    half = img_size // 2
    fbp = fbp[cy - half: cy + half, cx - half: cx + half]
         
    fbp = fbp - fbp.min()
    fbp = fbp / (fbp.max() + 1e-8)
    return fbp.astype(np.float32)

