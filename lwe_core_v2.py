"""
lwe_core_v2.py  ── LWE 核心模組（三種雜訊分布版）
支援：均勻 U(-k,k)、離散高斯 DG(sigma)、中心二項 CBD(eta)
"""
import numpy as np
import time
from scipy import stats

# ─────────────────────────────────────────────────────────
# 三種採樣器
# ─────────────────────────────────────────────────────────

def sample_uniform(k, size, rng):
    return rng.integers(-k, k + 1, size=size)

def sample_gaussian(sigma, size, rng, truncate=6):
    bound = int(np.ceil(truncate * sigma)) + 1
    n_total = size if isinstance(size, int) else int(np.prod(size))
    result = np.zeros(n_total, dtype=np.int64)
    filled = 0
    while filled < n_total:
        needed = n_total - filled
        candidates = np.round(rng.normal(0, sigma, size=needed * 2)).astype(np.int64)
        mask = np.abs(candidates) <= bound
        accepted = candidates[mask][: needed - filled]
        result[filled : filled + len(accepted)] = accepted
        filled += len(accepted)
    return result.reshape(size) if isinstance(size, tuple) else result

def sample_cbd(eta, size, rng):
    n_total = size if isinstance(size, int) else int(np.prod(size))
    a = rng.integers(0, 2, size=(n_total, eta)).sum(axis=1)
    b = rng.integers(0, 2, size=(n_total, eta)).sum(axis=1)
    result = (a - b).astype(np.int64)
    return result.reshape(size) if isinstance(size, tuple) else result

def sample_noise(dist, param, size, rng):
    if dist == "uniform":
        return sample_uniform(param, size, rng)
    elif dist == "gaussian":
        return sample_gaussian(param, size, rng)
    elif dist == "cbd":
        return sample_cbd(param, size, rng)
    raise ValueError(f"未知分布：{dist}")

# ─────────────────────────────────────────────────────────
# 有效雜訊比 ρ_eff
# ─────────────────────────────────────────────────────────

def rho_eff(dist, param, q, m):
    if dist == "uniform":
        sigma_r = param * np.sqrt(m / 6)
    elif dist == "gaussian":
        sigma_r = param * np.sqrt(m / 2)
    elif dist == "cbd":
        sigma_r = np.sqrt(m * param / 4)
    else:
        return 0.0
    return sigma_r / (q / 4)

def theory_pfail(rho):
    if rho <= 0:
        return 0.0
    return float(np.clip(2 * stats.norm.cdf(-1.0 / rho), 0, 1))

# ─────────────────────────────────────────────────────────
# 金鑰生成 / 加密 / 解密
# ─────────────────────────────────────────────────────────

def keygen(n, q, m, dist, param, rng):
    s = rng.integers(0, q, size=n)
    A = rng.integers(0, q, size=(m, n))
    e = sample_noise(dist, param, m, rng)
    b = (A @ s + e) % q
    return s, A, b

def encrypt(A, b, q, mu, rng, min_weight_ratio=0.25):
    m = len(b)
    min_w = max(1, int(m * min_weight_ratio))
    for _ in range(10000):
        r = rng.integers(0, 2, size=m)
        if r.sum() >= min_w:
            break
    u = (r @ A) % q
    v = int((r @ b + (q // 2) * mu) % q)
    return u, v

def decrypt(s, u, v, q):
    d = int((v - int(u @ s)) % q)
    dist0 = min(d, q - d)
    disth = min(abs(d - q // 2), q - abs(d - q // 2))
    return 1 if disth < dist0 else 0

# ─────────────────────────────────────────────────────────
# 批次錯誤率
# ─────────────────────────────────────────────────────────

def measure_error_rate(n, q, m, dist, param, trials=1000, seed=42):
    rng = np.random.default_rng(seed)
    errors = 0
    for _ in range(trials):
        s, A, b = keygen(n, q, m, dist, param, rng)
        mu = int(rng.integers(0, 2))
        u, v = encrypt(A, b, q, mu, rng)
        if decrypt(s, u, v, q) != mu:
            errors += 1
    reff = rho_eff(dist, param, q, m)
    return {
        "dist": dist, "param": param,
        "n": n, "q": q, "m": m,
        "rho_eff": reff,
        "theory_pfail": theory_pfail(reff),
        "error_rate": errors / trials,
        "error_count": errors,
        "trials": trials,
    }

# ─────────────────────────────────────────────────────────
# UTF-8 文字加解密
# ─────────────────────────────────────────────────────────

def encrypt_text(n, q, m, dist, param, text, encoding="utf-8", seed=42):
    rng = np.random.default_rng(seed)
    s, A, b = keygen(n, q, m, dist, param, rng)
    raw = text.encode(encoding)
    bits = []
    for byte in raw:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    t0 = time.perf_counter()
    cts = []
    for bit in bits:
        u, v = encrypt(A, b, q, bit, rng)
        cts.append((u, v))
    enc_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    recovered_bits = [decrypt(s, u, v, q) for u, v in cts]
    dec_time = time.perf_counter() - t0
    while len(recovered_bits) % 8 != 0:
        recovered_bits.append(0)
    raw_out = bytearray()
    for i in range(0, len(recovered_bits), 8):
        byte_val = int("".join(str(b) for b in recovered_bits[i:i+8]), 2)
        raw_out.append(byte_val)
    try:
        recovered = raw_out[: len(raw)].decode(encoding, errors="replace")
    except Exception:
        recovered = ""
    return {
        "text": text,
        "recovered": recovered,
        "correct": recovered == text,
        "bit_count": len(bits),
        "encrypt_count": len(bits),
        "encrypt_ms": enc_time * 1000,
        "decrypt_ms": dec_time * 1000,
        "rho_eff": rho_eff(dist, param, q, m),
    }
