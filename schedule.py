

import numpy as np

from lipschitz.oracle import estimate_lipschitz_rectangular

__all__ = ["CurriculumScheduler"]

class CurriculumScheduler:

    def __init__(self,
                 net,
                 T,
                 kp_min=2.0,
                 kp_max_init=50.0,
                 oracle_alpha=1.0,
                 oracle_ema_iters=5,
                 oracle_percentile=99.9,
                 freq_ratio_start=0.2,
                 k_gamma=1.0,
                 k_v=1.0,
                 k_out=1.0,
                 k_enc=4.0,
                 k_min_body=1.0,
                 eta=0.3,
                 bm3d_sigma=0.01):
        self.net = net
        self.T = int(T)

        self.kp_min           = float(kp_min)
        self.oracle_alpha     = float(oracle_alpha)
        self.oracle_ema_iters = int(oracle_ema_iters)
        self.oracle_percentile = float(oracle_percentile)
        self.freq_ratio_start = float(freq_ratio_start)
        self.k_gamma          = float(k_gamma)
        self.k_v              = float(k_v)
        self.k_out            = float(k_out)
        self.k_enc            = float(k_enc)
        self.k_min_body       = float(k_min_body)

        self.eta         = float(eta)
        self.bm3d_sigma  = float(bm3d_sigma)

        self._kp_max        = float(kp_max_init)
        self._ema_count     = 0
        self._oracle_frozen = False

    def step(self, iter_idx, prior):
                                                                  
        if not self._oracle_frozen:
            L_hat = self._estimate_kp_max(prior)
            scaled = self.oracle_alpha * L_hat
            if self._ema_count == 0:
                self._kp_max = scaled
            else:
                self._kp_max = 0.9 * self._kp_max + 0.1 * scaled
                                                        
            self._kp_max = max(self._kp_max, self.kp_min * 1.1)
            self._ema_count += 1
            if self._ema_count >= self.oracle_ema_iters:
                self._oracle_frozen = True

        r = iter_idx / max(self.T - 1, 1)

        freq_ratio = self.freq_ratio_start + (1.0 - self.freq_ratio_start) * r
        self.net.rff.set_frequency_window(freq_ratio)
        lip_rff = self.net.rff.lipschitz()

        kp_t = self.kp_min + (self._kp_max - self.kp_min) * r

        denom  = max(lip_rff * self.k_gamma * self.k_out, 1e-6)
        k_body = max(kp_t / denom, self.k_min_body)

        self.net.encoder.set_lipschitz(self.k_enc)
        self.net.mlp.set_budgets(
            k_gamma=self.k_gamma,
            k_v=self.k_v,
            k_body=k_body,
            k_out=self.k_out,
            k_min=self.k_min_body,
        )

        bgt = self.net.report_budgets()
        info = {
            'iter_idx'      : iter_idx,
            'r'             : round(r, 4),
            'freq_ratio'    : round(freq_ratio, 4),
            'lip_rff'       : round(lip_rff, 4),
            'kp_target'     : round(kp_t, 4),
            'kp_max'        : round(self._kp_max, 4),
            'k_body'        : round(k_body, 4),
            'oracle_frozen' : self._oracle_frozen,
            'K_p'           : round(bgt['K_p'], 4) if bgt['K_p'] else None,
            'K_z'           : round(bgt['K_z'], 4) if bgt['K_z'] else None,
            'eta'           : self.eta,
            'bm3d_sigma'    : self.bm3d_sigma,
        }
        return self.eta, self.bm3d_sigma, info

    def _estimate_kp_max(self, prior):

        prior_np = np.asarray(prior, dtype=np.float64)
                                                                     
        if prior_np.max() > 1.0:
            prior_np = prior_np / prior_np.max()
                                                              
        _, _, _, grad_mag = estimate_lipschitz_rectangular(
            prior_np, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
        return float(np.percentile(grad_mag, self.oracle_percentile))

    def reset(self):

        self._ema_count     = 0
        self._oracle_frozen = False

    def set_kp_max(self, value):

        self._kp_max        = float(value)
        self._oracle_frozen = True

    def __repr__(self):
        return (f"CurriculumScheduler(T={self.T}, "
                f"kp=[{self.kp_min:.1f}->{self._kp_max:.1f}], "
                f"freq=[{self.freq_ratio_start:.2f}->1.0], "
                f"eta={self.eta}, oracle_frozen={self._oracle_frozen})")

if __name__ == "__main__":
    import torch
    from FLIC import FLICNet

    torch.manual_seed(0)
    img_size = 64
    grid = 2 * int(img_size * 2 ** 0.5)

    net = FLICNet(grid_size=grid, feature_dim=16,
                    rff_num_features=32, mlp_width=64, mlp_depth=2)

    scheduler = CurriculumScheduler(net=net, T=20,
                                    kp_min=2.0, kp_max_init=20.0,
                                    oracle_ema_iters=3, eta=0.3)
    print(scheduler)

    prior = np.random.rand(img_size, img_size).astype(np.float64)
    for i in range(5):
        eta, sig, info = scheduler.step(i, prior)
        print(f"iter {i:2d} | r={info['r']:.2f} | freq={info['freq_ratio']:.2f} "
              f"| K_p={info['K_p']} | k_body={info['k_body']:.3f} "
              f"| oracle_frozen={info['oracle_frozen']}")

    print("smoke test OK")