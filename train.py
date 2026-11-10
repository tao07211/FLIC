import os
import SimpleITK as sitk
import numpy as np
import torch
import dataset
import commentjson as json
from torch.utils import data
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils import *
from skimage.transform import radon, iradon
import argparse
import FLIC
import sampling
import schedule as sched_mod
from bm3d import bm3d
import torch.nn.functional as F
import yaml

def eval_model(model, val_loader, test_loader, prior_img, device):
    model.eval()
    img_size = prior_img.shape[0]
    D = int(img_size * (2 ** 0.5))
    prior_tensor = torch.FloatTensor(prior_img).unsqueeze(0).unsqueeze(0).to(device)
    recon = None

    with torch.no_grad():
        for i, (ray_sample) in enumerate(val_loader):
            ray_sample = ray_sample.to(device).float()
            pre_intensity = model(prior_tensor, ray_sample).view(-1, 2 * D, 2 * D, 1)
            _pre_img = pre_intensity[0].squeeze().float().cpu().numpy()
            _pre_img = _pre_img[(2 * D - img_size) // 2: (2 * D + img_size) // 2,
                                (2 * D - img_size) // 2: (2 * D + img_size) // 2]
            print("recon range:", _pre_img.min(), _pre_img.max())
            _pre_img = np.clip(_pre_img, 0, 1)
        recon = _pre_img

    return recon

def train_w_sino(model, prior_img, lr, train_epoch, train_loader,
                 val_loader, test_loader, device, dv_fbp, result_path,
                 freq_ratio_start=0.2, k_body_min=1.0,
                 oracle_alpha=1.0, oracle_percentile=99.9,
                 slice_name='slice'):
    img_size = prior_img.shape[0]
    D = int(img_size * (2 ** 0.5))
    loss_fun = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loop = tqdm(range(train_epoch), colour='green', ncols=120)
    prior_tensor = torch.FloatTensor(prior_img).unsqueeze(0).unsqueeze(0).to(device)
    eval_interval = 50                    

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    import csv
    csv_path = os.path.join(result_path, f'{slice_name}_metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'psnr', 'ssim', 'loss', 'freq', 'k_body'])

    best_psnr = -1.0
    best_recon = None

    prior_np = np.asarray(prior_img, dtype=np.float64)
    if prior_np.max() > 1.0:
        prior_np = prior_np / prior_np.max()
    from lipschitz.oracle import estimate_lipschitz_rectangular
    _, _, _, grad_mag = estimate_lipschitz_rectangular(
        prior_np, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
    L_hat = float(np.percentile(grad_mag, oracle_percentile))
    L_median = float(np.median(grad_mag[grad_mag > 1e-8]))
    k_body_max = max(oracle_alpha * (L_hat / L_median), k_body_min * 1.5)
    k_body_max = min(k_body_max, 20.0)
    print(f"[Oracle] L_hat={L_hat:.2f}  L_median={L_median:.2f}  "
          f"k_body_max={k_body_max:.2f}")

    for e in train_loop:
        r = e / max(train_epoch - 1, 1)
        ratio = freq_ratio_start + (1.0 - freq_ratio_start) * r
        model.rff.set_frequency_window(ratio)
        r_body = min(r / 0.8, 1.0)
        k_body_t = k_body_min + (k_body_max - k_body_min) * r_body
        model.mlp.set_budgets(
            k_gamma=1.0, k_v=1.0,
            k_body=k_body_t,
            k_out=1.0, k_min=1.0,
        )

        model.train()
        loss_train = 0
        for _, (ray, weights, proj_l) in enumerate(train_loader):
            ray     = ray.to(device).float()
            weights = weights.to(device).float()
            proj_l  = proj_l.to(device).float()
            pre_intensity = model(prior_tensor, ray)
            pre_proj_l = sampling.forward_project(pre_intensity, weights)
            loss = loss_fun(pre_proj_l, proj_l.to(pre_proj_l.dtype))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        avg_loss = loss_train / len(train_loader)
        train_loop.set_description(
            f'[{e+1}/{train_epoch}], lr:{optimizer.param_groups[0]["lr"]:.6f}'
            f' Loss:{avg_loss:.6f} freq:{ratio:.2f} k_body:{k_body_t:.2f}')

        if (e + 1) % eval_interval == 0 or e == train_epoch - 1:
            recon = eval_model(model, val_loader, test_loader, prior_img, device)
            psnr, ssim = cal_psnr(recon, dv_fbp), cal_ssim(recon, dv_fbp)

            csv_writer.writerow([e+1, f'{psnr:.4f}', f'{ssim:.6f}',
                                 f'{avg_loss:.6f}', f'{ratio:.4f}', f'{k_body_t:.4f}'])
            csv_file.flush()

            print(f'Epoch {e+1},  PSNR/SSIM: {psnr:.2f}/{ssim:.4f}'
                  f'  freq:{ratio:.2f}  k_body:{k_body_t:.2f}')

            if psnr > best_psnr:
                best_psnr = psnr
                best_ssim = ssim
                best_recon = recon.copy()
                best_epoch = e + 1

    csv_file.close()
    print(f'[{slice_name}] Best: Epoch {best_epoch}  '
          f'PSNR={best_psnr:.2f}  SSIM={best_ssim:.4f}')

    best_fname = f'{slice_name}_psnr{best_psnr:.2f}_ssim{best_ssim:.4f}.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(best_recon),
                    os.path.join(result_path, best_fname))

    return model, best_recon

def train_w_reg(model, prior_img, reg_img, lr, train_epoch, train_loader,
                val_loader, test_loader, device, dv_fbp, result_path, lamda):
    img_size = prior_img.shape[0]
    D = int(img_size * (2 ** 0.5))
    loss_fun = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loop = tqdm(range(train_epoch), colour='green', ncols=120)
    prior_tensor = torch.FloatTensor(prior_img).unsqueeze(0).unsqueeze(0).to(device)
    reg_tensor = generate_prior_image_for_fanbeam(prior_img=reg_img, D=D, device=device)
    summary_epoch = train_epoch // 5
    train_coord = dataset.build_coordinate_val(D, 0)
    train_coord = torch.tensor(train_coord, dtype=torch.float32).to(device)
    patch_size = 128
    pad = (2 * D - img_size) // 2                        

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for e in train_loop:
        model.train()
        loss_sino = 0
        loss_reg  = 0
        for _, (ray, weights, proj_l) in enumerate(train_loader):
            batch_size, sample_N = ray.shape[0], ray.shape[1]
            ray     = ray.to(device).float()
            weights = weights.to(device).float()
            proj_l  = proj_l.to(device).float()

            pre_intensity = model(prior_tensor, ray)
            pre_proj_l = sampling.forward_project(pre_intensity, weights)
            sino_loss = loss_fun(pre_proj_l, proj_l.to(pre_proj_l.dtype))

            start_x = np.random.randint(pad, pad + img_size - patch_size)
            start_y = np.random.randint(pad, pad + img_size - patch_size)

            select_coord = train_coord[:,
                                       start_x: start_x + patch_size,
                                       start_y: start_y + patch_size]

            select_intensity = reg_tensor[:, :,
                                           start_x: start_x + patch_size,
                                           start_y: start_y + patch_size].flatten()

            pre_reg = model(prior_tensor, select_coord).flatten()
            reg_loss = loss_fun(pre_reg, select_intensity)

            loss = sino_loss + lamda * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sino += sino_loss.item()
            loss_reg  += reg_loss.item()

        train_loop.set_description(
            f'[{e+1}/{train_epoch}], lr:{optimizer.param_groups[0]["lr"]:.6f}'
            f' Sino:{loss_sino/len(train_loader):.6f}'
            f' Reg:{loss_reg/len(train_loader):.6f}')

        if (e + 1) % summary_epoch == 0 or e == train_epoch - 1:
            recon = eval_model(model, val_loader, test_loader, prior_img, device)
            psnr, ssim = cal_psnr(recon, dv_fbp), cal_ssim(recon, dv_fbp)
            print(f'Epoch {e+1},  PSNR/SSIM: {psnr:.2f}/{ssim:.4f}')
            sitk.WriteImage(sitk.GetImageFromArray(recon),
                            os.path.join(result_path, f'recon_{e+1}.nii.gz'))
            torch.save(model.state_dict(),
                       os.path.join(result_path, f'model_{e+1}.pth'))

    return model, recon

def train(config):
                    
    input_path  = config["file"]["input_path"]
    result_path = config["file"]["result_path"]
    num_sv, num_dv, img_size = (config["file"]["num_sv"],
                                config["file"]["num_dv"],
                                config["file"]["img_size"])
    I0      = config["file"]["I0"]
    lr      = config["train"]["lr"]
    sample_N = config["train"]["sample_N"]
    gpu     = config['train']['gpu']
    lamda   = config['train']['lamda']
    iter_num = config['train']['iter_num']

    spacing   = config['file']['fan_geometry']['spacing']
    det_count = config['file']['fan_geometry']['det_count']
    D         = config['file']['fan_geometry']['SOD']

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    gt_img = sitk.GetArrayFromImage(sitk.ReadImage(input_path))
    if img_size != gt_img.shape[0]:
        raise ValueError('GT Image size does not match the input image size')

    sensor_pos = np.linspace(-det_count / 2, det_count / 2, det_count) * spacing
    fan_pose   = np.rad2deg(np.arctan(sensor_pos / (2 * D)))
    L = len(fan_pose)

    projection_angle_num = num_dv
    scale = int(num_dv / num_sv)
    dv_sino = fanbeam_projection(gt_img, D, det_count, spacing, projection_angle_num, device)
    sv_sino = dv_sino[::scale, :]
    det_count = dv_sino.shape[1]
    sensor_pos = np.linspace(-det_count / 2, det_count / 2, det_count) * spacing
    fan_pose   = np.rad2deg(np.arctan(sensor_pos / (2 * D)))
    L = det_count
    
    if I0 is not None:
        taskname = f'SV_{num_sv}_I0_{I0:.0e}/'
        sv_sino  = add_noise_to_sino(sv_sino, I0, 0.01)
    else:
        taskname = f'SV_{num_sv}/'

    result_path = os.path.join(result_path, taskname)
    os.makedirs(result_path, exist_ok=True)
    from skimage.transform import radon as sk_radon, iradon as sk_iradon
    _angles_dv = np.linspace(0, 180, num_dv, endpoint=False)
    _angles_sv = _angles_dv[::scale]
    _sk_sino   = sk_radon(gt_img, theta=_angles_dv, circle=False)                 
    _sk_sv     = _sk_sino[:, ::scale]                                              
    sv_fbp_recon = sk_iradon(_sk_sv, theta=_angles_sv,
                              filter_name='ramp', circle=False)
    _h, _w = sv_fbp_recon.shape
    _cy, _cx = _h // 2, _w // 2
    _half = img_size // 2
    sv_fbp_recon = sv_fbp_recon[_cy-_half:_cy+_half, _cx-_half:_cx+_half]
    sv_fbp_recon = sv_fbp_recon - sv_fbp_recon.min()
    sv_fbp_recon = sv_fbp_recon / (sv_fbp_recon.max() + 1e-8)
    sv_fbp_recon = sv_fbp_recon.astype(np.float32)

    if sv_fbp_recon.shape != (img_size, img_size):
        from skimage.transform import resize
        sv_fbp_recon = resize(sv_fbp_recon, (img_size, img_size),
                              order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)

    psnr, ssim = cal_psnr(sv_fbp_recon, gt_img), cal_ssim(sv_fbp_recon, gt_img)
    print("gt_img range:", gt_img.min(), gt_img.max())
    print("sv_fbp_recon range:", sv_fbp_recon.min(), sv_fbp_recon.max())
    print("dv_sino range:", dv_sino.min(), dv_sino.max())
    print("sv_sino range:", sv_sino.min(), sv_sino.max())
    print("gt_img dtype:", gt_img.dtype)
    print("x tensor range:", gt_img.min(), gt_img.max())
    print(f'SV_FBP  PSNR/SSIM: {psnr:.2f}/{ssim:.4f}')

    _angles_dv_full = np.linspace(0, 180, num_dv, endpoint=False)
    _sk_sino_dv = sk_radon(gt_img, theta=_angles_dv_full, circle=False)
    dv_fbp_recon = sk_iradon(_sk_sino_dv, theta=_angles_dv_full,
                              filter_name='ramp', circle=False)
    _h, _w = dv_fbp_recon.shape
    _cy, _cx = _h // 2, _w // 2
    _half = img_size // 2
    dv_fbp_recon = dv_fbp_recon[_cy-_half:_cy+_half, _cx-_half:_cx+_half]
    dv_fbp_recon = dv_fbp_recon - dv_fbp_recon.min()
    dv_fbp_recon = dv_fbp_recon / (dv_fbp_recon.max() + 1e-8)
    dv_fbp_recon = dv_fbp_recon.astype(np.float32)
    psnr_dv = cal_psnr(dv_fbp_recon, gt_img)
    print(f"Dense-view FBP PSNR: {psnr_dv:.2f}")

    sitk.WriteImage(sitk.GetImageFromArray(sv_fbp_recon),
                    os.path.join(result_path, 'sv_fbp.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(gt_img),
                    os.path.join(result_path, 'gt_img.nii.gz'))

    train_set = dataset.TrainData(
        sin_data=sv_sino, theta=num_sv,
        sample_N=sample_N, fan_pose=fan_pose, D=D)
    train_loader = data.DataLoader(
        train_set, batch_size=config['train']['batch_size'], shuffle=True)

    val_loader = data.DataLoader(
        dataset.ValData(D=D, angle=0), batch_size=1, shuffle=False)
    test_loader = data.DataLoader(
        dataset.TestData(theta=projection_angle_num, D=D, fan_pose=fan_pose),
        batch_size=6, shuffle=False)

    grid_size = 2 * int(img_size * (2 ** 0.5))
    En_SCOPE = FLIC.FLICNet(
        grid_size=grid_size,
        **config.get('network_rbl', {})
    ).to(device)

    schedule_config = dict(config.get('schedule', {}))
    k_body_min = schedule_config.pop('k_body_min', 1.0)
    k_body_max = schedule_config.pop('k_body_max', 5.0)
    
    scheduler = sched_mod.CurriculumScheduler(
        net=En_SCOPE,
        T=iter_num,
        **schedule_config                      
    )
    print(scheduler)

    sv_fbp_recon = np.clip(sv_fbp_recon, 0, 1)
                                                     
    slice_name = os.path.splitext(
                 os.path.splitext(os.path.basename(input_path))[0]
             )[0]

    model, recon = train_w_sino(
        model=En_SCOPE, prior_img=sv_fbp_recon,
        lr=lr, train_epoch=2500,
        train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, device=device,
        dv_fbp=gt_img,
        freq_ratio_start=config.get('schedule', {}).get('freq_ratio_start', 0.2),
        k_body_min=k_body_min,
        oracle_alpha=0.5,
        slice_name=slice_name,
        result_path=result_path + '1_iter/')

    for iternum in range(iter_num):
                   
        eta, bm3d_sig, info = scheduler.step(iternum, prior=recon)
        recon = bm3d(recon, sigma_psd=bm3d_sig)
        sitk.WriteImage(sitk.GetImageFromArray(recon),
                        os.path.join(result_path,
                                     f'denoised_recon_{iternum+1}.nii.gz'))
        psnr, ssim = cal_psnr(recon, gt_img), cal_ssim(recon, gt_img)
        print(f'Iter {iternum+1}, denoised PSNR/SSIM: {psnr:.2f}/{ssim:.4f}')

        print(f'Iter {iternum+1} schedule | '
              f'freq={info["freq_ratio"]:.2f} '
              f'K_p={info["K_p"]} k_body={info["k_body"]:.3f} '
              f'kp_max={info["kp_max"]:.2f} eta={eta:.2f}')

        train_loader.dataset.resample(recon, eta=eta)

        model, recon = train_w_reg(
            model=model, prior_img=sv_fbp_recon, reg_img=recon,
            lr=lr / 10, train_epoch=500,
            train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, device=device,
            dv_fbp=gt_img,
            result_path=result_path + f'{iternum+2}_iter/',
            lamda=lamda,
            lamda_tv=config['train'].get('lamda_tv', 0.1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FLIC')
    parser.add_argument('--config', type=str,
                        default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = json.load(f)

    input_path = config["file"]["input_path"]

    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        import glob
        files = sorted(glob.glob(os.path.join(input_path, '*.nii.gz')))
        if len(files) == 0:
            raise FileNotFoundError(f"No .nii.gz in {input_path}")
    else:
        raise ValueError(f"Path not found: {input_path}")

    print(f"Processing {len(files)} slice(s)")

    result_base = config["file"]["result_path"]
    import csv
    summary_path = os.path.join(result_base, 'summary.csv')
    os.makedirs(result_base, exist_ok=True)
    summary_file = open(summary_path, 'w', newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(['slice', 'psnr', 'ssim'])

    for fpath in files:
        slice_name = os.path.splitext(
                         os.path.splitext(os.path.basename(fpath))[0]
                     )[0]
        print(f'\n{"="*60}')
        print(f'Processing: {slice_name}')
        print(f'{"="*60}')

        config["file"]["input_path"] = fpath
        config["file"]["result_path"] = os.path.join(result_base, slice_name)

        train(config)

        num_sv = config["file"]["num_sv"]
        I0 = config["file"].get("I0", None)          
        
        if I0 is not None:
            taskname = f'SV_{num_sv}_I0_{I0:.0e}'
        else:
            taskname = f'SV_{num_sv}'

        iter1_csv = os.path.join(config["file"]["result_path"],
                                 taskname,                                               
                                 '1_iter',
                                 f'{slice_name}_metrics.csv')

        if os.path.exists(iter1_csv):
            import pandas as pd
            df = pd.read_csv(iter1_csv)
            best_row = df.loc[df['psnr'].astype(float).idxmax()]
            summary_writer.writerow([slice_name,
                                     f'{float(best_row["psnr"]):.4f}',
                                     f'{float(best_row["ssim"]):.6f}'])
            summary_file.flush()
        else:
                                         
            print(f"[Warning] 找不到子指标文件，请检查路径: {iter1_csv}")

    summary_file.close()
    print(f'\nAll done. Summary saved to {summary_path}')