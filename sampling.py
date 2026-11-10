

import math
import numpy as np
import torch

__all__ = [
    "compute_gradient_map",
    "build_ray_cdfs",
    "resample_from_prior",
    "forward_project",
]

def compute_gradient_map(prior, grid_size):

    prior = np.asarray(prior, dtype=np.float64)
    H, W = prior.shape

    gy, gx = np.gradient(prior, edge_order=2)
    G_img = np.sqrt(gx ** 2 + gy ** 2)

    G_padded = np.zeros((grid_size, grid_size), dtype=np.float64)
    pt = (grid_size - H) // 2
    pl = (grid_size - W) // 2
    G_padded[pt: pt + H, pl: pl + W] = G_img
    return G_padded

def _interp_grid(G, pts, gs):

    x, y = pts[:, 0], pts[:, 1]
    row_f = np.clip((x + 1.0) / 2.0 * (gs - 1), 0.0, gs - 1.0 - 1e-7)
    col_f = np.clip((y + 1.0) / 2.0 * (gs - 1), 0.0, gs - 1.0 - 1e-7)

    r0 = row_f.astype(np.int32)
    c0 = col_f.astype(np.int32)
    r1 = np.minimum(r0 + 1, gs - 1)
    c1 = np.minimum(c0 + 1, gs - 1)
    dr = row_f - r0
    dc = col_f - c0

    return ((1 - dr) * (1 - dc) * G[r0, c0] +
            (1 - dr) *      dc  * G[r0, c1] +
                 dr  * (1 - dc) * G[r1, c0] +
                 dr  *      dc  * G[r1, c1])

def build_ray_cdfs(rays_geom, G_padded, grid_size, eps=1e-6):

    na, L, N_u, _ = rays_geom.shape

    pts_flat = rays_geom.reshape(-1, 2)
    g_flat   = _interp_grid(G_padded, pts_flat, grid_size)

    g_mat   = g_flat.reshape(na * L, N_u) + eps
    pdf_mat = g_mat / g_mat.sum(axis=1, keepdims=True)
    cdfs    = np.cumsum(pdf_mat, axis=1)
    return cdfs.reshape(na, L, N_u)

def _sample_t_from_cdf(cdf, N_u, N, eta, rng):

    n_unif = max(1, int(round(eta * N)))
    n_pdf  = N - n_unif
    t_nodes = np.linspace(0.0, 1.0, N_u)

    t_unif = rng.uniform(0.0, 1.0, n_unif)

    if n_pdf <= 0:
        return t_unif

    u  = rng.uniform(0.0, 1.0, n_pdf)
    k  = np.searchsorted(cdf, u, side="left").clip(0, N_u - 1)

    cdf_lo = np.where(k == 0, 0.0,  cdf[np.maximum(k - 1, 0)])
    cdf_hi = cdf[k]
    t_lo   = np.where(k == 0, 0.0,  t_nodes[np.maximum(k - 1, 0)])
    t_hi   = t_nodes[k]
    delta  = np.maximum(cdf_hi - cdf_lo, 1e-12)
    t_pdf  = t_lo + (u - cdf_lo) / delta * (t_hi - t_lo)

    return np.concatenate([t_unif, t_pdf])

def _compute_weights(t_sorted):

    N = len(t_sorted)
    if N == 1:
        return np.array([1.0])
    mids       = np.empty(N + 1)
    mids[0]    = 0.0
    mids[1:-1] = 0.5 * (t_sorted[:-1] + t_sorted[1:])
    mids[-1]   = 1.0
    return (mids[1:] - mids[:-1]) * N 

def resample_from_prior(prior, rays_geom, grid_size, N, eta,
                        N_b=1, eps=1e-6, seed=None):
    rng = np.random.default_rng(seed)
    na, L, N_u, _ = rays_geom.shape

    G_padded = compute_gradient_map(prior, grid_size)
    cdfs = build_ray_cdfs(rays_geom, G_padded, grid_size, eps=eps)

    if N_b > 1:
        mid   = N_u // 2
        mids  = rays_geom[:, :, mid, :]                            
        delta = np.zeros_like(mids)
        delta[:, 1:-1] = (mids[:, 2:] - mids[:, :-2]) / 2.0
        delta[:, 0]    = mids[:, 1] - mids[:, 0]
        delta[:, -1]   = mids[:, -1] - mids[:, -2]
                                                                           
        frac = (np.arange(N_b, dtype=np.float64) / max(N_b - 1, 1) - 0.5)
                                                       
        perp_offsets = delta[:, :, None, :] * frac[None, None, :, None]
    else:
        perp_offsets = None

    if N_b > 1:
        new_rays    = np.empty((na, L, N_b, N, 2), dtype=np.float64)
        new_weights = np.empty((na, L, N_b, N),    dtype=np.float64)
    else:
        new_rays    = np.empty((na, L, N, 2), dtype=np.float64)
        new_weights = np.empty((na, L, N),    dtype=np.float64)

    for a in range(na):
        for l in range(L):
            near_c = rays_geom[a, l,  0]         
            far_c  = rays_geom[a, l, -1]         
            cdf    = cdfs[a, l]                     

            for b in range(N_b):
                if perp_offsets is not None:
                    off  = perp_offsets[a, l, b]
                    near = near_c + off
                    far  = far_c  + off
                else:
                    near, far = near_c, far_c

                t_vals   = _sample_t_from_cdf(cdf, N_u, N, eta, rng)
                t_vals   = np.clip(t_vals, 0.0, 1.0)
                t_sorted = np.sort(t_vals)

                pts = near[None, :] + t_sorted[:, None] * (far - near)[None, :]
                wts = _compute_weights(t_sorted)

                if N_b > 1:
                    new_rays[a, l, b]    = pts
                    new_weights[a, l, b] = wts
                else:
                    new_rays[a, l]    = pts
                    new_weights[a, l] = wts

    return new_rays, new_weights

def forward_project(intensities_flat, weights):
    mu   = intensities_flat.squeeze(-1).view(weights.shape)
    proj = (mu * weights).sum(dim=-1)                                       
    if proj.dim() > 2:
        proj = proj.mean(dim=-1)                                             
    return proj

if __name__ == "__main__":
    import torch

    rng_np   = np.random.default_rng(0)
    img_size = 64
    D        = int(img_size * math.sqrt(2))
    gs       = 2 * D
    N        = gs
    na, L    = 6, 32

    t_uni     = np.linspace(0, 1, N)
    nears     = rng_np.uniform(-0.8, 0.8, (na, L, 2))
    fars      = rng_np.uniform(-0.8, 0.8, (na, L, 2))
    rays_geom = (nears[:, :, None, :] +
                 t_uni[None, None, :, None] * (fars - nears)[:, :, None, :])

    prior = rng_np.random((img_size, img_size))

    rays1, w1 = resample_from_prior(prior, rays_geom, gs, N,
                                    eta=0.4, N_b=1, seed=42)
    print("N_b=1  rays    :", rays1.shape)
    print("N_b=1  weights :", w1.shape)
    assert w1.shape == (na, L, N)
    assert np.allclose(w1.sum(-1), 1.0, atol=1e-9), "weights must sum to 1"

    rays3, w3 = resample_from_prior(prior, rays_geom, gs, N,
                                    eta=0.4, N_b=3, seed=42)
    print("N_b=3  rays    :", rays3.shape)
    print("N_b=3  weights :", w3.shape)
    assert w3.shape == (na, L, 3, N)

    batch, sN = 2, 8
    wt1  = torch.from_numpy(w1[:batch, :sN]).float()               
    mu1  = torch.rand(batch * sN * N, 1)
    p1   = forward_project(mu1, wt1)
    print("forward_project N_b=1 :", p1.shape, "  (expect (2,8))")
    assert p1.shape == (batch, sN)

    wt3  = torch.from_numpy(w3[:batch, :sN]).float()                  
    mu3  = torch.rand(batch * sN * 3 * N, 1)
    p3   = forward_project(mu3, wt3)
    print("forward_project N_b=3 :", p3.shape, "  (expect (2,8))")
    assert p3.shape == (batch, sN)

    print("smoke test OK")