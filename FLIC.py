

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import RFFEncoder, LipschitzImageEncoder, LipschitzMLP

__all__ = ["FLICNet"]

class FLICNet(nn.Module):

    def __init__(self,
                 grid_size,
                 feature_dim=32,
                 encoder_hidden=48,
                 rff_num_features=256,
                 rff_sigma=20.0,
                 rff_transition=0.15,
                 rff_seed=None,
                 mlp_width=128,
                 mlp_depth=3,
                 mlp_body_type="spectral",
                 mlp_allocation="exponential",
                 n_power_iterations=1):
        super().__init__()
        self.grid_size = int(grid_size)
        self.feature_dim = int(feature_dim)

        self.encoder = LipschitzImageEncoder(
            out_dim=feature_dim,
            hidden=encoder_hidden,
            n_power_iterations=n_power_iterations,
        )

        self.rff = RFFEncoder(
            in_dim=2,
            num_features=rff_num_features,
            sigma=rff_sigma,
            transition=rff_transition,
            seed=rff_seed,
        )

        self.mlp = LipschitzMLP(
            feat_dim=feature_dim,
            rff_dim=self.rff.out_dim,
            width=mlp_width,
            depth=mlp_depth,
            body_type=mlp_body_type,
            allocation=mlp_allocation,
            n_power_iterations=n_power_iterations,
        )

    def _sample_features(self, feat, xy_flat):

        _, C, H, W = feat.shape
        g = self.grid_size
        assert g >= H and g >= W, "grid_size must be >= feature-map size"

        if H != g or W != g:
            pad_l = (g - W) // 2
            pad_r = g - W - pad_l
            pad_t = (g - H) // 2
            pad_b = g - H - pad_t
            feat = F.pad(feat, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)

        grid = xy_flat.unsqueeze(0).unsqueeze(0)                      
        sampled = F.grid_sample(feat, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
                                           
        return sampled.permute(0, 2, 3, 1).reshape(-1, C)

    def forward(self, img, xy):

        assert img.dim() == 4 and img.shape[0] == 1, \
            "prior image must have shape (1, 1, H, W)"

        feat = self.encoder(img)                                            
        xy_flat = xy.reshape(-1, 2)                            

        V = self._sample_features(feat, xy_flat)                      
        gamma = self.rff(xy_flat)                                    
        mu = self.mlp(V, gamma)                                
        return mu

    def set_iteration_budgets(self, freq_ratio, k_enc,
                              k_gamma, k_v, k_body, k_out, k_min=1.0):

        self.rff.set_frequency_window(freq_ratio)
        self.encoder.set_lipschitz(k_enc)
        self.mlp.set_budgets(k_gamma, k_v, k_body, k_out, k_min)

    def report_budgets(self):

        lip_rff = self.rff.lipschitz()
        kp_mlp = self.mlp.path_product("coord")
        kz_mlp = self.mlp.path_product("image")
        k_enc = getattr(self.encoder, "_k_total", None)

        out = {"lip_rff": lip_rff, "k_enc": k_enc,
               "kp_mlp_side": kp_mlp, "kz_mlp_side": kz_mlp,
               "K_p": None, "K_z": None}
        if kp_mlp is not None:
            out["K_p"] = lip_rff * kp_mlp
        if kz_mlp is not None and k_enc is not None:
            out["K_z"] = k_enc * kz_mlp
        return out

if __name__ == "__main__":
    torch.manual_seed(0)

    img_size = 64
    grid = 2 * int(img_size * 2 ** 0.5)                               

    net = FLICNet(
        grid_size=grid,
        feature_dim=32,
        rff_num_features=64,
        rff_sigma=20.0,
        mlp_width=128,
        mlp_depth=3,
        mlp_body_type="spectral",
        mlp_allocation="exponential",
    )

    net.set_iteration_budgets(freq_ratio=0.5, k_enc=4.0,
                              k_gamma=2.0, k_v=1.0, k_body=8.0, k_out=1.0)

    img = torch.rand(1, 1, img_size, img_size)

    xy4 = torch.rand(2, 16, grid, 2) * 2.0 - 1.0
    mu4 = net(img, xy4)
    print("xy (4-D)   :", tuple(xy4.shape))
    print("mu         :", tuple(mu4.shape), " (expect (2*16*grid, 1))")

    xy5 = torch.rand(2, 16, 3, grid, 2) * 2.0 - 1.0
    mu5 = net(img, xy5)
    print("xy (5-D)   :", tuple(xy5.shape))
    print("mu         :", tuple(mu5.shape), " (forward is shape-agnostic)")

    print("budgets    :", net.report_budgets())
    print("smoke test OK")
