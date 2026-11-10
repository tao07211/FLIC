import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipschitz.spectral_norm import spectral_norm, SpectralNorm
from lipschitz.sll_layers import KLipschitzSDPBasedLipschitzLinearLayer
from lipschitz.budget import get_allocation_function

__all__ = [
    "make_klip_linear",
    "set_layer_lipschitz",
    "RFFEncoder",
    "LipschitzImageEncoder",
    "LipschitzMLP",
]

def make_klip_linear(in_dim, out_dim, k=1.0, n_power_iterations=1, bias=True):

    layer = nn.Linear(in_dim, out_dim, bias=bias)
    spectral_norm(layer, name="weight",
                  n_power_iterations=n_power_iterations,
                  lipschitz_const=float(k))
    return layer

def set_layer_lipschitz(layer, k):

    k = float(k)
    if isinstance(layer, KLipschitzSDPBasedLipschitzLinearLayer):
        layer.k_coef_lip = k
        return
    for hook in layer._forward_pre_hooks.values():
        if isinstance(hook, SpectralNorm):
            hook.lipschitz_const = k
            return
    raise RuntimeError(
        f"set_layer_lipschitz: no SpectralNorm hook found on {type(layer).__name__}"
    )

class RFFEncoder(nn.Module):
    def __init__(self, in_dim=2, num_features=256, sigma=20.0,
                 transition=0.15, seed=None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_features = int(num_features)
        self.transition = float(transition)

        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(int(seed))
        B = torch.randn(self.num_features, self.in_dim, generator=gen) * float(sigma)

        order = torch.argsort(B.norm(dim=1))
        B = B[order].contiguous()

        self.register_buffer("B", B)                                           
        self.register_buffer("mask", torch.ones(self.num_features))        

    @property
    def out_dim(self):
        return 2 * self.num_features

    def set_frequency_window(self, ratio):

        ratio = float(min(max(ratio, 0.0), 1.0))
        m = self.num_features
        rank = torch.arange(m, device=self.B.device, dtype=self.B.dtype)
        rank = rank / max(m - 1, 1)
        mask = (ratio - rank) / max(self.transition, 1e-6) + 1.0
        self.mask = mask.clamp_(0.0, 1.0)

    def lipschitz(self):

        w = (self.mask ** 2).unsqueeze(1)                        
        M = (self.B * w).t() @ self.B                                      
        lam_max = torch.linalg.eigvalsh(M)[-1].clamp_min(0.0)
        return float(2.0 * math.pi * torch.sqrt(lam_max))

    def forward(self, xy):
                                                          
        proj = 2.0 * math.pi * F.linear(xy, self.B)              
        cos = torch.cos(proj) * self.mask
        sin = torch.sin(proj) * self.mask
        return torch.cat([cos, sin], dim=-1)                      

class LipschitzImageEncoder(nn.Module):

    def __init__(self, out_dim=32, hidden=48, n_power_iterations=1):
        super().__init__()
        c1 = nn.Conv2d(1, hidden, kernel_size=3, stride=1, padding=1)
        c2 = nn.Conv2d(hidden, out_dim, kernel_size=3, stride=1, padding=1)
        spectral_norm(c1, name="weight",
                      n_power_iterations=n_power_iterations, lipschitz_const=1.0)
        spectral_norm(c2, name="weight",
                      n_power_iterations=n_power_iterations, lipschitz_const=1.0)
        self.conv1 = c1
        self.conv2 = c2
        self.act = nn.ReLU(inplace=True)
        self.out_dim = int(out_dim)
        self._k_total = None

    def set_lipschitz(self, k_total):

        per = float(k_total) ** 0.5
        set_layer_lipschitz(self.conv1, per)
        set_layer_lipschitz(self.conv2, per)
        self._k_total = float(k_total)

    def forward(self, img):
                                                 
        return self.conv2(self.act(self.conv1(img)))

class LipschitzMLP(nn.Module):

    def __init__(self, feat_dim, rff_dim, width=128, depth=3,
                 body_type="spectral", allocation="exponential",
                 n_power_iterations=1):
        super().__init__()
        assert body_type in ("spectral", "sll")
        if body_type == "sll":
            assert feat_dim is not None
        self.body_type = body_type
        self.width = int(width)
        self.depth = int(depth)
        self.allocation_name = allocation
        self.alloc_fn = get_allocation_function(allocation)
        self._npi = int(n_power_iterations)

        self.proj_gamma = make_klip_linear(rff_dim, width, k=1.0,
                                           n_power_iterations=n_power_iterations)
        self.proj_v = make_klip_linear(feat_dim, width, k=1.0,
                                       n_power_iterations=n_power_iterations)

        self.body = nn.ModuleList()
        for _ in range(self.depth):
            if body_type == "spectral":
                self.body.append(
                    make_klip_linear(width, width, k=1.0,
                                     n_power_iterations=n_power_iterations)
                )
            else:                                                  
                self.body.append(
                    KLipschitzSDPBasedLipschitzLinearLayer(width, width,
                                                           k_coef_lip=1.0)
                )

        self.head = make_klip_linear(width, 1, k=1.0,
                                     n_power_iterations=n_power_iterations)

        self.act = nn.ReLU(inplace=True)
        self.out_act = nn.Softplus(beta=10)
        self._budgets = {}                                                    

    def set_budgets(self, k_gamma, k_v, k_body, k_out, k_min=1.0):
        set_layer_lipschitz(self.proj_gamma, k_gamma)
        set_layer_lipschitz(self.proj_v, k_v)
        set_layer_lipschitz(self.head, k_out)

        if self.depth > 0:

            result = self.alloc_fn(float(k_body), self.depth, float(k_min))
            per_layer = list(result[0])
            for layer, k_l in zip(self.body, per_layer):
                set_layer_lipschitz(layer, k_l)
        else:
            per_layer = []

        self._budgets = dict(
            k_gamma=float(k_gamma), k_v=float(k_v),
            k_body=float(k_body), k_out=float(k_out), k_min=float(k_min),
            body_layers=[float(x) for x in per_layer],
        )

    def path_product(self, path):

        if not self._budgets:
            return None
        body_prod = 1.0
        for x in self._budgets["body_layers"]:
            body_prod *= x
        shared = body_prod * self._budgets["k_out"]
        if path == "coord":
            return self._budgets["k_gamma"] * shared
        if path == "image":
            return self._budgets["k_v"] * shared
        raise ValueError("path must be 'coord' or 'image'")

    def forward(self, V, gamma):
                                                                  
        h = self.proj_gamma(gamma) + self.proj_v(V)
        if self.body_type == "spectral":
            for layer in self.body:
                h = layer(self.act(h))
        else:                                                    
            for layer in self.body:
                h = layer(h)
        return self.out_act(self.head(self.act(h)))

if __name__ == "__main__":
    torch.manual_seed(0)
    feat_dim, num_freq = 32, 64
    N = 1024

    rff = RFFEncoder(in_dim=2, num_features=num_freq, sigma=20.0, seed=0)
    enc = LipschitzImageEncoder(out_dim=feat_dim)
    mlp = LipschitzMLP(feat_dim=feat_dim, rff_dim=rff.out_dim,
                       width=128, depth=3, body_type="spectral",
                       allocation="exponential")

    rff.set_frequency_window(ratio=0.5)
    enc.set_lipschitz(k_total=4.0)
    mlp.set_budgets(k_gamma=2.0, k_v=1.0, k_body=8.0, k_out=1.0, k_min=1.0)

    img = torch.rand(1, 1, 64, 64)
    xy = torch.rand(N, 2) * 2.0 - 1.0

    feat = enc(img)                                                         
    V = feat.permute(0, 2, 3, 1).reshape(-1, feat_dim)[:N]                             
    gamma = rff(xy)                                                   
    mu = mlp(V, gamma)                                       

    print("feature map :", tuple(feat.shape))
    print("gamma        :", tuple(gamma.shape), "  out_dim =", rff.out_dim)
    print("mu           :", tuple(mu.shape))
    print("Lip(RFF)     : %.4f" % rff.lipschitz())
    print("K^p (MLP side):", mlp.path_product("coord"))
    print("K^z (MLP side):", mlp.path_product("image"))
    print("smoke test OK")