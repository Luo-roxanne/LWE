"""
lwe_core.py
─────────────────────────────────────────────────────────────────────────────
LWE（帶錯誤學習問題）核心模組

包含：
  - 參數驗證
  - 金鑰生成
  - 加密 / 解密（單 bit）
  - 批次正確率測試
  - 暴力搜尋攻擊
  - BKZ 攻擊介面（需安裝 fpylll）

數學符號對應：
  n  維度（秘密向量維度）
  q  模數（質數）
  k  雜訊幅度（每個雜訊分量 ∈ [-k, k] 均勻分布）
  m  公鑰行數（建議 m = 2n）
  s  秘密向量 ∈ ℤq^n（私鑰）
  A  公開矩陣 ∈ ℤq^(m×n)
  e  雜訊向量 ∈ ℤq^m
  b  公鑰向量：b = As + e (mod q)
  r  加密用隨機位元向量 ∈ {0,1}^m
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LWEPublicKey:
    A: np.ndarray   # shape: (m, n)
    b: np.ndarray   # shape: (m,)
    n: int
    q: int
    k: int
    m: int

@dataclass
class LWEPrivateKey:
    s: np.ndarray   # shape: (n,)
    n: int
    q: int

@dataclass
class LWECiphertext:
    u: np.ndarray   # shape: (n,)  = rᵀA mod q
    v: int          # scalar       = rᵀb + ⌊q/2⌋μ mod q

@dataclass
class AttackResult:
    success: bool
    elapsed: float          # 秒
    recovered_s: Optional[np.ndarray]
    method: str
    attempts: int


# ─────────────────────────────────────────────────────────────────────────────
# 輔助函數
# ─────────────────────────────────────────────────────────────────────────────

def is_prime(n: int) -> bool:
    """簡單質數判斷（實驗用，夠快）"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def validate_params(n: int, q: int, k: int, m: int) -> str:
    """
    驗證 LWE 參數合法性
    回傳空字串表示合法，否則回傳錯誤訊息
    """
    if n < 1:
        return f"維度 n={n} 必須 ≥ 1"
    if not is_prime(q):
        return f"模數 q={q} 必須是質數"
    if k < 1:
        return f"雜訊幅度 k={k} 必須 ≥ 1"
    if m < n:
        return f"公鑰行數 m={m} 建議 ≥ n={n}（目前 m < n 會降低安全性）"
    if 2 * k >= q // 4:
        # 雜訊比 ρ = k / (q/4) > 0.5，解密可能不穩定
        rho = k / (q / 4)
        return (f"警告：雜訊比 ρ = k/(q/4) = {rho:.2f} ≥ 0.5，"
                f"解密可能失敗。建議降低 k 或增大 q。")
    return ""


def noise_ratio(k: int, q: int) -> float:
    """
    雜訊比 ρ = k / (q/4)
    ρ < 1  → 解密安全區
    ρ ≈ 1  → 相變點
    ρ > 1  → 解密大量失敗
    """
    return k / (q / 4)


def theoretical_fail_rate(k: int, q: int, m: int) -> float:
    """
    理論解密失敗率估計（常態近似）
    
    推導：rᵀe 的分布近似 N(0, σ²)
    其中 σ² = k² * w / 3，w = m/2（r 的期望漢明重量）
    解密失敗條件：|rᵀe| ≥ ⌊q/4⌋
    
    P_fail ≈ 2 * Φ(-⌊q/4⌋ / σ)
    """
    from scipy import stats
    w = m / 2                        # r 的期望漢明重量
    sigma = np.sqrt(k**2 * w / 3)   # rᵀe 的標準差
    threshold = q // 4
    if sigma == 0:
        return 0.0
    p_fail = 2 * stats.norm.cdf(-threshold / sigma)
    return float(np.clip(p_fail, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 金鑰生成
# ─────────────────────────────────────────────────────────────────────────────

def keygen(n: int, q: int, k: int, m: int,
           rng: Optional[np.random.Generator] = None
           ) -> tuple[LWEPublicKey, LWEPrivateKey]:
    """
    LWE 金鑰生成
    
    金鑰生成流程：
      1. 隨機選取私鑰 s ∈ ℤq^n
      2. 隨機選取公開矩陣 A ∈ ℤq^(m×n)
      3. 從均勻分布 U[-k, k] 採樣雜訊向量 e ∈ ℤm
      4. 計算公鑰向量 b = As + e (mod q)
    """
    if rng is None:
        rng = np.random.default_rng()

    s = rng.integers(0, q, size=n)                   # 私鑰
    A = rng.integers(0, q, size=(m, n))              # 公開矩陣
    e = rng.integers(-k, k + 1, size=m)              # 雜訊（均勻分布）
    b = (A @ s + e) % q                              # 公鑰向量

    pub = LWEPublicKey(A=A, b=b, n=n, q=q, k=k, m=m)
    priv = LWEPrivateKey(s=s, n=n, q=q)
    return pub, priv


# ─────────────────────────────────────────────────────────────────────────────
# 加密 / 解密
# ─────────────────────────────────────────────────────────────────────────────

def encrypt(pub: LWEPublicKey, mu: int,
            rng: Optional[np.random.Generator] = None,
            min_weight_ratio: float = 0.25
            ) -> LWECiphertext:
    """
    LWE 加密（單 bit，μ ∈ {0, 1}）
    
    流程：
      1. 隨機選取 r ∈ {0,1}^m，要求漢明重量 ≥ m * min_weight_ratio
         （排除全零向量及過於稀疏的 r，確保加密強度）
      2. u = rᵀA  (mod q)
      3. v = rᵀb + ⌊q/2⌋ * μ  (mod q)
    """
    if mu not in (0, 1):
        raise ValueError(f"明文 μ 必須是 0 或 1，收到 {mu}")
    if rng is None:
        rng = np.random.default_rng()

    min_weight = max(1, int(pub.m * min_weight_ratio))
    half_q = pub.q // 2

    # 不斷重試直到 r 的漢明重量夠大
    for _ in range(10000):
        r = rng.integers(0, 2, size=pub.m)
        if np.sum(r) >= min_weight:
            break
    else:
        raise RuntimeError("無法生成有效的 r（請增大 m 或降低 min_weight_ratio）")

    u = (r @ pub.A) % pub.q
    v = int((r @ pub.b + half_q * mu) % pub.q)
    return LWECiphertext(u=u, v=v)


def decrypt(priv: LWEPrivateKey, ct: LWECiphertext) -> int:
    """
    LWE 解密
    
    流程：
      1. d = v - u·s  (mod q)
      2. 若 d 比 ⌊q/2⌋ 更接近（mod q 意義下），回傳 1；否則回傳 0
    
    正確性保證：
      d = v - u·s = rᵀe + ⌊q/2⌋·μ  (mod q)
      當 |rᵀe| < ⌊q/4⌋ 時，解密正確
    """
    q = priv.q
    d = int((ct.v - int(ct.u @ priv.s)) % q)

    # 判斷 d 更靠近 0 還是 ⌊q/2⌋（在模 q 的環上）
    half_q = q // 2
    dist_to_0    = min(d, q - d)
    dist_to_half = min(abs(d - half_q), q - abs(d - half_q))

    return 1 if dist_to_half < dist_to_0 else 0


# ─────────────────────────────────────────────────────────────────────────────
# 批次正確率測試
# ─────────────────────────────────────────────────────────────────────────────

def measure_error_rate(n: int, q: int, k: int, m: int,
                       trials: int = 1000,
                       seed: Optional[int] = None) -> dict:
    """
    執行 trials 次獨立加解密，統計解密錯誤率
    
    每次使用新的金鑰對，確保統計獨立性
    
    回傳：
      error_rate    : 解密錯誤率（0.0 ~ 1.0）
      error_count   : 錯誤次數
      trials        : 測試次數
      rho           : 雜訊比 ρ = k / (q/4)
      theory_fail   : 理論失敗率（常態近似）
    """
    rng = np.random.default_rng(seed)
    errors = 0

    for _ in range(trials):
        pub, priv = keygen(n, q, k, m, rng=rng)
        mu = rng.integers(0, 2)
        ct = encrypt(pub, int(mu), rng=rng)
        mu_hat = decrypt(priv, ct)
        if mu_hat != mu:
            errors += 1

    rho = noise_ratio(k, q)
    try:
        theory = theoretical_fail_rate(k, q, m)
    except ImportError:
        theory = None

    return {
        "error_rate":   errors / trials,
        "error_count":  errors,
        "trials":       trials,
        "rho":          rho,
        "theory_fail":  theory,
        "n": n, "q": q, "k": k, "m": m,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 暴力搜尋攻擊
# ─────────────────────────────────────────────────────────────────────────────

def brute_force_attack(pub: LWEPublicKey,
                       timeout: float = 60.0) -> AttackResult:
    """
    暴力搜尋攻擊：窮舉所有可能的 s ∈ ℤq^n
    
    複雜度：O(q^n)，僅適用於 n ≤ 10 的低維情況
    
    判斷標準：對於候選秘密向量 s_cand，計算殘差
      r_i = (b_i - a_i · s_cand) mod q
    若所有殘差的絕對值（mod q 意義下）均 ≤ k，
    則認定 s_cand 為正確的秘密向量
    """
    n, q, k, m = pub.n, pub.q, pub.k, pub.m
    A, b = pub.A, pub.b

    start = time.time()
    attempts = 0

    # 使用 numpy 的多維索引產生器
    # 對 q^n 個候選向量逐一測試
    from itertools import product

    for s_cand_tuple in product(range(q), repeat=n):
        elapsed = time.time() - start
        if elapsed > timeout:
            return AttackResult(
                success=False,
                elapsed=elapsed,
                recovered_s=None,
                method="brute_force",
                attempts=attempts,
            )

        attempts += 1
        s_cand = np.array(s_cand_tuple, dtype=np.int64)

        # 計算殘差向量
        residuals = (b - A @ s_cand) % q
        # 將殘差映射到 [-q/2, q/2] 範圍
        residuals_centered = np.where(residuals > q // 2,
                                      residuals - q,
                                      residuals)

        # 若所有殘差的絕對值 ≤ k，視為找到正確的 s
        if np.all(np.abs(residuals_centered) <= k):
            elapsed = time.time() - start
            return AttackResult(
                success=True,
                elapsed=elapsed,
                recovered_s=s_cand,
                method="brute_force",
                attempts=attempts,
            )

    elapsed = time.time() - start
    return AttackResult(
        success=False,
        elapsed=elapsed,
        recovered_s=None,
        method="brute_force",
        attempts=attempts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BKZ 攻擊介面（需安裝 fpylll）
# ─────────────────────────────────────────────────────────────────────────────

def bkz_attack(pub: LWEPublicKey,
               block_size: int = 10,
               timeout: float = 60.0) -> AttackResult:
    """
    BKZ 格基約化攻擊
    
    將 LWE 問題轉化為格上的最近向量問題（CVP）：
    
      構造嵌入格（Kannan's embedding）：
        [  A  |  I_m  |  0  ]
        [  b  |  0    |  1  ]
      目標：在此格中找最短向量，對應 (e, -1) 這個短向量
    
    需要安裝 fpylll：pip install fpylll
    
    參數：
      block_size  BKZ 的區塊大小（越大越強但越慢，建議 10-20）
      timeout     超時限制（秒）
    """
    try:
        from fpylll import IntegerMatrix, BKZ, LLL
        from fpylll.algorithms.bkz2 import BKZReduction
    except ImportError:
        return AttackResult(
            success=False,
            elapsed=0.0,
            recovered_s=None,
            method="bkz_unavailable",
            attempts=0,
        )

    start = time.time()
    n, q, m = pub.n, pub.q, pub.m
    A, b = pub.A, pub.b

    # 構造 (m+1) × (m+n+1) 的嵌入格矩陣
    # 格式：[ q*I_m  0   0 ]
    #        [  A    I_n  0 ]
    #        [  b    0    1 ]
    dim = m + n + 1
    M = IntegerMatrix(m + 1, dim)

    for i in range(m):
        M[i, i] = q
    for i in range(m):
        for j in range(n):
            M[m, i] = int(b[i])
        for j in range(n):
            M[i + m - m, m + j] = int(A[i, j])
    M[m, dim - 1] = 1

    # 執行 LLL 預處理 + BKZ
    try:
        LLL.reduction(M)
        BKZ.reduction(M, BKZ.Param(block_size=block_size))
    except Exception:
        elapsed = time.time() - start
        return AttackResult(
            success=False,
            elapsed=elapsed,
            recovered_s=None,
            method="bkz_error",
            attempts=1,
        )

    elapsed = time.time() - start

    # 在 BKZ 縮減後的格中找對應 (e, s) 的行
    for i in range(M.nrows):
        row = [M[i][j] for j in range(dim)]
        if row[dim - 1] in (1, -1):
            sign = row[dim - 1]
            s_cand = np.array([sign * row[m + j] % q for j in range(n)],
                              dtype=np.int64)
            residuals = (b - A @ s_cand) % q
            residuals_centered = np.where(residuals > q // 2,
                                          residuals - q, residuals)
            if np.all(np.abs(residuals_centered) <= pub.k):
                return AttackResult(
                    success=True,
                    elapsed=elapsed,
                    recovered_s=s_cand,
                    method="bkz",
                    attempts=1,
                )

    return AttackResult(
        success=False,
        elapsed=elapsed,
        recovered_s=None,
        method="bkz",
        attempts=1,
    )
