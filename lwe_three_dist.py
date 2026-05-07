"""
lwe_three_dist.py
三種雜訊分布的 LWE 安全性比較研究
分布：均勻 U(-k,k)、離散高斯 DG(sigma)、中心二項 CBD(eta)
"""

import numpy as np
import json, csv, os, time
from scipy import stats
from itertools import product

os.makedirs("results3", exist_ok=True)

# ─────────────────────────────────────────────────────────
# 採樣器
# ─────────────────────────────────────────────────────────

def sample_uniform(k, size, rng):
    """均勻分布 U(-k, k)"""
    return rng.integers(-k, k+1, size=size)

def sample_gaussian(sigma, size, rng, truncate=6):
    """
    離散高斯分布 DG(sigma)
    採用拒絕採樣：先從連續高斯採樣，四捨五入後截斷到 [-truncate*sigma, truncate*sigma]
    """
    bound = int(np.ceil(truncate * sigma)) + 1
    result = np.zeros(size if isinstance(size, int) else np.prod(size), dtype=np.int64)
    n_total = result.size
    filled = 0
    while filled < n_total:
        needed = n_total - filled
        # 過採樣以減少迭代次數
        candidates = rng.normal(0, sigma, size=needed * 2)
        candidates = np.round(candidates).astype(np.int64)
        # 截斷
        mask = np.abs(candidates) <= bound
        accepted = candidates[mask][:needed]
        n_accept = len(accepted)
        result[filled:filled+n_accept] = accepted
        filled += n_accept
    if isinstance(size, tuple):
        return result.reshape(size)
    return result

def sample_cbd(eta, size, rng):
    """
    中心二項分布 CBD(eta)
    eᵢ = sum(a_bits) - sum(b_bits)，a_bits, b_bits 各為 eta 個獨立均勻位元
    方差精確為 eta/2
    """
    if isinstance(size, int):
        total = size
        shape = (size,)
    else:
        total = np.prod(size)
        shape = size
    a = rng.integers(0, 2, size=(total, eta)).sum(axis=1)
    b = rng.integers(0, 2, size=(total, eta)).sum(axis=1)
    return (a - b).reshape(shape)

# ─────────────────────────────────────────────────────────
# 有效雜訊比 ρ_eff = σ_r / (q/4)
# σ_r = rᵀe 的標準差
# ─────────────────────────────────────────────────────────

def rho_eff_uniform(k, q, m):
    sigma_r = k * np.sqrt(m / 6)
    return sigma_r / (q / 4)

def rho_eff_gaussian(sigma_g, q, m):
    sigma_r = sigma_g * np.sqrt(m / 2)
    return sigma_r / (q / 4)

def rho_eff_cbd(eta, q, m):
    sigma_r = np.sqrt(m * eta / 4)
    return sigma_r / (q / 4)

def theory_pfail(rho_eff):
    """理論解密失敗率（共通公式）P_fail ≈ 2Φ(-1/ρ_eff)"""
    if rho_eff <= 0:
        return 0.0
    return float(np.clip(2 * stats.norm.cdf(-1.0 / rho_eff), 0, 1))

# ─────────────────────────────────────────────────────────
# LWE 加解密（支援三種分布）
# ─────────────────────────────────────────────────────────

def keygen(n, q, m, dist, param, rng):
    s = rng.integers(0, q, size=n)
    A = rng.integers(0, q, size=(m, n))
    if dist == "uniform":
        e = sample_uniform(param, m, rng)
    elif dist == "gaussian":
        e = sample_gaussian(param, m, rng)
    elif dist == "cbd":
        e = sample_cbd(param, m, rng)
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
    disth = min(abs(d - q//2), q - abs(d - q//2))
    return 1 if disth < dist0 else 0

def measure_error_rate(n, q, m, dist, param, trials=1000, seed=42):
    rng = np.random.default_rng(seed)
    errors = 0
    for _ in range(trials):
        s, A, b = keygen(n, q, m, dist, param, rng)
        mu = int(rng.integers(0, 2))
        u, v = encrypt(A, b, q, mu, rng)
        mu_hat = decrypt(s, u, v, q)
        if mu_hat != mu:
            errors += 1
    reff = {
        "uniform":  rho_eff_uniform(param, q, m),
        "gaussian": rho_eff_gaussian(param, q, m),
        "cbd":      rho_eff_cbd(param, q, m),
    }[dist]
    return {
        "dist": dist, "param": param, "n": n, "q": q, "m": m,
        "rho_eff": reff,
        "theory_pfail": theory_pfail(reff),
        "error_rate": errors / trials,
        "error_count": errors,
        "trials": trials,
    }

# ─────────────────────────────────────────────────────────
# 實驗 A：相變點比較（固定 n=8, q=101, m=16）
# 以 ρ_eff 為橫軸，三種分布並排比較
# ─────────────────────────────────────────────────────────

def exp_A(trials=1000, seed=42):
    print("\n" + "="*60)
    print("實驗A：三種分布的相變點比較")
    print("固定 n=8, q=101, m=16，掃描 ρ_eff 從 0.05 到 1.2")
    print("="*60)

    N, Q, M = 8, 101, 16
    # 目標 ρ_eff 序列
    rho_targets = [round(x, 3) for x in np.arange(0.05, 1.25, 0.05)]
    results = []

    for rho_t in rho_targets:
        row = {"rho_eff_target": rho_t}

        # 均勻分布：k = round(ρ_eff × q/4 / √(m/6))
        k = max(1, round(rho_t * (Q/4) / np.sqrt(M/6)))
        r1 = measure_error_rate(N, Q, M, "uniform", k, trials=trials, seed=seed+int(rho_t*100))
        row["uniform_k"] = k
        row["uniform_rho_eff"] = r1["rho_eff"]
        row["uniform_error"] = r1["error_rate"]
        row["uniform_theory"] = r1["theory_pfail"]

        # 離散高斯：sigma_G = ρ_eff × q/4 / √(m/2)
        sg = max(0.5, rho_t * (Q/4) / np.sqrt(M/2))
        r2 = measure_error_rate(N, Q, M, "gaussian", sg, trials=trials, seed=seed+int(rho_t*100)+1000)
        row["gauss_sigma"] = round(sg, 3)
        row["gauss_rho_eff"] = r2["rho_eff"]
        row["gauss_error"] = r2["error_rate"]
        row["gauss_theory"] = r2["theory_pfail"]

        # CBD：eta = round(4 × (ρ_eff × q/4)² / m)，至少 1
        eta = max(1, round(4 * (rho_t * Q/4)**2 / M))
        r3 = measure_error_rate(N, Q, M, "cbd", eta, trials=trials, seed=seed+int(rho_t*100)+2000)
        row["cbd_eta"] = eta
        row["cbd_rho_eff"] = r3["rho_eff"]
        row["cbd_error"] = r3["error_rate"]
        row["cbd_theory"] = r3["theory_pfail"]

        results.append(row)

        print(f"ρ_eff≈{rho_t:.2f} | "
              f"均勻(k={k}): {r1['error_rate']*100:.1f}% | "
              f"高斯(σ={sg:.1f}): {r2['error_rate']*100:.1f}% | "
              f"CBD(η={eta}): {r3['error_rate']*100:.1f}%")

    # 存檔
    with open("results3/expA_phase_transition.json","w",encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open("results3/expA_phase_transition.csv","w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    print("✓ 實驗A 完成，結果存至 results3/expA_*")
    return results

# ─────────────────────────────────────────────────────────
# 實驗 B：安全性地圖（固定 n=8, t=1, m=16）
# 對每種分布掃描多個 q 值和雜訊參數，建立三張熱圖數據
# ─────────────────────────────────────────────────────────

def sieve(limit):
    is_p = [True]*(limit+1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(limit**0.5)+1):
        if is_p[i]:
            for j in range(i*i, limit+1, i): is_p[j] = False
    return [i for i in range(17, limit+1) if is_p[i]]

def exp_B(q_max=127, trials=500, seed=42):
    print("\n" + "="*60)
    print("實驗B：三種分布的安全性地圖")
    print(f"固定 n=8, m=16，掃描 q ∈ 質數[17,{q_max}]")
    print("="*60)

    N, M = 8, 16
    q_list = sieve(q_max)
    results = {"uniform": [], "gaussian": [], "cbd": []}

    for qi, q in enumerate(q_list):
        print(f"\n  q={q}:")

        # 均勻分布：掃描 k = 1 到 q//4+5
        for k in range(1, q//4+6):
            reff = rho_eff_uniform(k, q, M)
            r = measure_error_rate(N, q, M, "uniform", k, trials=trials,
                                   seed=seed+qi*1000+k)
            results["uniform"].append({
                "q": q, "param": k, "param_name": "k",
                "rho_eff": reff, "theory_pfail": theory_pfail(reff),
                "error_rate": r["error_rate"]
            })

        # 高斯分布：掃描 sigma_G = 0.5, 1.0, 1.5, ..., q//8+1
        sg_max = max(2.0, (q/4) / np.sqrt(M/2) * 1.2)
        sg_list = [round(x, 1) for x in np.arange(0.5, sg_max, 0.5)]
        for sg in sg_list:
            reff = rho_eff_gaussian(sg, q, M)
            r = measure_error_rate(N, q, M, "gaussian", sg, trials=trials,
                                   seed=seed+qi*1000+int(sg*10)+500)
            results["gaussian"].append({
                "q": q, "param": sg, "param_name": "sigma",
                "rho_eff": reff, "theory_pfail": theory_pfail(reff),
                "error_rate": r["error_rate"]
            })

        # CBD：掃描 eta = 1, 2, 3, ..., eta_max
        eta_max = max(4, int(4 * (q/4)**2 / M * 1.3) + 1)
        for eta in range(1, min(eta_max, 20)+1):
            reff = rho_eff_cbd(eta, q, M)
            r = measure_error_rate(N, q, M, "cbd", eta, trials=trials,
                                   seed=seed+qi*1000+eta+800)
            results["cbd"].append({
                "q": q, "param": eta, "param_name": "eta",
                "rho_eff": reff, "theory_pfail": theory_pfail(reff),
                "error_rate": r["error_rate"]
            })

        print(f"    均勻: {len([r for r in results['uniform'] if r['q']==q])} 筆 | "
              f"高斯: {len([r for r in results['gaussian'] if r['q']==q])} 筆 | "
              f"CBD: {len([r for r in results['cbd'] if r['q']==q])} 筆")

    # 存檔
    for dist, data in results.items():
        with open(f"results3/expB_map_{dist}.json","w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with open(f"results3/expB_map_{dist}.csv","w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader(); writer.writerows(data)

    total = sum(len(v) for v in results.values())
    print(f"\n✓ 實驗B 完成，共 {total} 筆數據，存至 results3/expB_map_*")
    return results

# ─────────────────────────────────────────────────────────
# 實驗 C：縮小版 Kyber（n=16, q=257, m=32）
# 三種分布各自搜尋最大安全雜訊參數（失敗率 < 0.1%）
# ─────────────────────────────────────────────────────────

def exp_C(trials=2000, seed=42):
    print("\n" + "="*60)
    print("實驗C：縮小版 Kyber 參數搜尋（n=16, q=257, m=32）")
    print("目標：錯誤率 < 0.1%（2000次中最多2次失敗）")
    print("="*60)

    N, Q, M = 16, 257, 32
    results = []

    # 均勻分布：掃描 k
    print("\n  【均勻分布】")
    best_k = None
    for k in range(1, Q//4+1):
        r = measure_error_rate(N, Q, M, "uniform", k, trials=trials, seed=seed+k)
        flag = "✓" if r["error_rate"] < 0.001 else "✗"
        if r["error_rate"] < 0.001:
            best_k = k
        print(f"    k={k:>3}, ρ_eff={r['rho_eff']:.4f}, 錯誤率={r['error_rate']*100:.3f}% {flag}")
        results.append({**r, "dist_label": "uniform", "target_met": r["error_rate"]<0.001})
        if r["error_rate"] >= 0.05:
            break
    print(f"  → 最大安全 k = {best_k}，ρ_eff = {rho_eff_uniform(best_k,Q,M):.4f}")

    # 高斯分布：掃描 sigma_G
    print("\n  【離散高斯分布】")
    best_sg = None
    for sg_10 in range(5, 100, 5):  # sigma 從 0.5 到 9.5，步長 0.5
        sg = sg_10 / 10
        r = measure_error_rate(N, Q, M, "gaussian", sg, trials=trials, seed=seed+sg_10+1000)
        flag = "✓" if r["error_rate"] < 0.001 else "✗"
        if r["error_rate"] < 0.001:
            best_sg = sg
        print(f"    σ={sg:.1f}, ρ_eff={r['rho_eff']:.4f}, 錯誤率={r['error_rate']*100:.3f}% {flag}")
        results.append({**r, "dist_label": "gaussian", "target_met": r["error_rate"]<0.001})
        if r["error_rate"] >= 0.05:
            break
    print(f"  → 最大安全 σ = {best_sg}，ρ_eff = {rho_eff_gaussian(best_sg,Q,M):.4f}")

    # CBD：掃描 eta
    print("\n  【CBD 分布】（η=3 對應 Kyber-512）")
    best_eta = None
    for eta in range(1, 20):
        r = measure_error_rate(N, Q, M, "cbd", eta, trials=trials, seed=seed+eta+2000)
        flag = "✓" if r["error_rate"] < 0.001 else "✗"
        kyber_mark = " ← Kyber-512 設定" if eta == 3 else ""
        if r["error_rate"] < 0.001:
            best_eta = eta
        print(f"    η={eta:>2}, ρ_eff={r['rho_eff']:.4f}, 錯誤率={r['error_rate']*100:.3f}% {flag}{kyber_mark}")
        results.append({**r, "dist_label": "cbd", "target_met": r["error_rate"]<0.001})
        if r["error_rate"] >= 0.05:
            break
    print(f"  → 最大安全 η = {best_eta}，ρ_eff = {rho_eff_cbd(best_eta,Q,M):.4f}")

    with open("results3/expC_kyber.json","w",encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open("results3/expC_kyber.csv","w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    print("\n✓ 實驗C 完成，結果存至 results3/expC_kyber.*")
    return results

# ─────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", help="A / B / C / all")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    args = parser.parse_args()

    trials_A = 300 if args.quick else 1000
    trials_B = 200 if args.quick else 500
    trials_C = 500 if args.quick else 2000
    q_max_B  = 71  if args.quick else 127

    t0 = time.time()
    if args.exp in ("A", "all"):
        exp_A(trials=trials_A)
    if args.exp in ("B", "all"):
        exp_B(q_max=q_max_B, trials=trials_B)
    if args.exp in ("C", "all"):
        exp_C(trials=trials_C)
    print(f"\n總耗時：{time.time()-t0:.1f}s")
    print(f"所有結果存於 results3/ 資料夾")
