import os
import glob
import pandas as pd

RESULT_BASE = "output/aapm_normal"                              
NUM_SV = 60                          
I0 = 1e6                                                     

if I0 is not None:
    taskname = f'SV_{NUM_SV}_I0_{I0:.0e}'
else:
    taskname = f'SV_{NUM_SV}'

all_psnr = []
all_ssim = []

slice_dirs = sorted(glob.glob(os.path.join(RESULT_BASE, "*")))

for sdir in slice_dirs:
    if not os.path.isdir(sdir):
        continue
    
    slice_name = os.path.basename(sdir)
    metrics_path = os.path.join(sdir, taskname, '1_iter', f'{slice_name}_metrics.csv')
    
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            if not df.empty:
                                  
                best_row = df.loc[df['psnr'].astype(float).idxmax()]
                all_psnr.append(float(best_row['psnr']))
                all_ssim.append(float(best_row['ssim']))
        except Exception:
            pass

if all_psnr:
    avg_psnr = sum(all_psnr) / len(all_psnr)
    avg_ssim = sum(all_ssim) / len(all_ssim)
    
    print("\n" + "="*35)
    print(f"统计切片总数: {len(all_psnr)} 个")
    print(f"👉 平均 PSNR: {avg_psnr:.4f}")
    print(f"👉 平均 SSIM: {avg_ssim:.6f}")
    print("="*35)
else:
    print(f"\n❌ 未提取到有效数据。请检查配置是否正确，或目标路径是否存在: \n{os.path.join(RESULT_BASE, '*', taskname)}")