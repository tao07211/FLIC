import numpy as np

def _scale_to_product(s_raw, k):

    s_raw = np.asarray(s_raw, dtype=float)
    assert np.all(s_raw > 0), "All s_raw must be positive."
    L = s_raw.size
    c = (k / np.prod(s_raw)) ** (1.0 / L)
    return s_raw * c

def _allocate_with_last_and_product(k, r, f):

    f = np.asarray(f, dtype=float)
    L = f.size
    assert L >= 1
    if L == 1:
        s = np.array([float(k)], dtype=float)
        return s, np.log(s)

    A = np.log(r)
    f_last = f[-1]
    sum_f = f.sum()

    denom = (sum_f - L * f_last)
                                                                                                                               
    if np.isclose(denom, 0.0):
                                                                                       
        x = (np.log(k) - A) / (L - 1)
        u = np.full(L, x, dtype=float)
        u[-1] = A
        s = np.exp(u)
        return s, u

    b = (np.log(k) - L * A) / denom
    a = A - b * f_last
    u = a + b * f
    s = np.exp(u)
    return s, u

def allocate_first(k, L, r=1.0):

    s = np.ones(L, dtype=float)
    s[0] = k
    u = np.log(s)
    return s, u

def allocate_uniform(k, L, r=1.0):

    logk = np.log(k)
    u = np.full(L, logk / L, dtype=float)
    s = np.exp(u)
    return s, u

def allocate_exponential(k, L, r=1.0):

    assert L >= 1
    if L == 1:
        s = np.array([float(k)], dtype=float)
        return s, np.log(s)

    v = np.linspace(1.0, 0.0, L)
    sum_v = v.sum()

    a = np.log(r)
    b = (np.log(k) - L * a) / sum_v

    u = a + b * v
    s = np.exp(u)
    return s, u

def allocate_linear(k, L, r, eps=1e-12):
                                                                         
    assert L >= 1 and k > 0 and r > 0
    if L == 1:
        s = np.array([float(k)]); return s, np.log(s)
    if L == 2:
        s = np.array([float(k)/float(r), float(r)]); return s, np.log(s)

    t = np.linspace(0.0, 1.0, L)                           
    logk = np.log(k)

    def logprod(s0):
        s = s0 + (r - s0) * t                                   
        return np.sum(np.log(np.maximum(s, eps)))                 

    lo, hi = eps, max(r, k**(1.0/L)) * 2.0               
    while logprod(hi) < logk: hi *= 2.0                               
    for _ in range(50):                                               
        mid = 0.5*(lo+hi)
        (lo if logprod(mid) < logk else hi).__class__                          
        if logprod(mid) < logk: lo = mid
        else: hi = mid

    s0 = 0.5*(lo+hi)
    s = s0 + (r - s0) * t
    return s, np.log(s)

def allocate_cosine_annealed(k, L, r, variant="front", tol=1e-12, max_it=80):

    assert k > 0 and r > 0 and L >= 2
    t = np.linspace(0.0, 1.0, L)                                
    if variant == "front":
        g = 0.5 * (1.0 + np.cos(np.pi * t))
    elif variant == "mid":
        g = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
    else:
        raise ValueError("variant must be 'front' or 'mid'.")

    g_front = g[:-1]                                       
    target = np.log(k / (r**L))

    g_pos = g_front[g_front > 0]
    if g_pos.size == 0:
                                                                         
        if not np.isclose(k, r**L):
            raise ValueError("No cosine leverage (all g=0) but k != r^L.")
        K = np.full(L, r, dtype=float)
        u = np.log(K)
        return K, u, 0.0

    alpha_lo = -1.0 / np.max(g_pos) + 1e-15                                       
    alpha_hi = 1.0
                                                                                       
    def F(a):
        return np.sum(np.log1p(a * g_front)) - target
    while F(alpha_hi) < 0.0:
        alpha_hi *= 2.0
        if alpha_hi > 1e12:
            raise RuntimeError("alpha_hi exploded; check k / r^L is not too large.")

    for _ in range(max_it):
        alpha_mid = 0.5 * (alpha_lo + alpha_hi)
        val = F(alpha_mid)
        if abs(val) < tol:
            alpha = alpha_mid
            break
        if val < 0:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid
    else:
        alpha = 0.5 * (alpha_lo + alpha_hi)

    K = np.empty(L, dtype=float)
    K[:-1] = r * (1.0 + alpha * g_front)
    K[-1]  = r
    u = np.log(K)
    return K, u

def get_allocation_function(name):
    name = name.lower()
    if name in ["uniform", "uni"]:
        return allocate_uniform
    elif name in ["allfirst", "first"]:
        return allocate_first
    elif name in ["exponential", "exp"]:
        return allocate_exponential
    elif name in ["linear", "lin"]:
        return allocate_linear
    elif name in ["cosine", "cos"]:
        return allocate_cosine_annealed
    else:
        raise ValueError(f"Unknown allocation function name: {name}")