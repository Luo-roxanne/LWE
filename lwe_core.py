"""
lwe_core.py  ── LWE 核心模組（完整版）
─────────────────────────────────────────────────────────────────────────────
涵蓋：
  金鑰生成、t-bit 加解密、批次錯誤率測試
  暴力搜尋攻擊、BKZ 攻擊介面
  UTF-8 / ASCII / Big5 字串加解密
  雜訊比 ρ、廣義雜訊比 ρ_t、理論失敗率

數學符號：
  n   維度（安全性主要來源）
  q   模數（質數，控制運算範圍）
  k   雜訊幅度，eᵢ ∈ U[-k, k]
  m   公鑰行數（建議 m = 2n）
  t   每次加密的 bit 數（t=1 為基本版）
  s   私鑰向量 ∈ ℤqⁿ
  A   公開矩陣 ∈ ℤqᵐˣⁿ（完全隨機）
  e   雜訊向量 ∈ ℤqᵐ
  b   公鑰向量：b = As + e (mod q)
  r   加密向量 ∈ {0,1}ᵐ，漢明重量 ≥ m/4
  ρ   雜訊比 = k / (q/4)（t=1 時的安全指標）
  ρ_t 廣義雜訊比 = k / (q / 2^(t+1))
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LWEParams:
    """LWE 系統參數"""
    n: int          # 維度
    q: int          # 模數（質數）
    k: int          # 雜訊幅度
    m: int          # 公鑰行數
    t: int = 1      # 每次加密 bit 數

    def __post_init__(self):
        err = validate_params(self.n, self.q, self.k, self.m, self.t)
        if err and not err.startswith("警告"):
            raise ValueError(err)

    @property
    def rho(self) -> float:
        """雜訊比 ρ = k / (q/4)，t=1 時的安全指標"""
        return noise_ratio(self.k, self.q, t=1)

    @property
    def rho_t(self) -> float:
        """廣義雜訊比 ρ_t = k / (q / 2^(t+1))"""
        return noise_ratio(self.k, self.q, t=self.t)

    @property
    def symbols_per_encrypt(self) -> int:
        """每次 LWE 加密可傳遞的符號數（2^t 個可能值）"""
        return 2 ** self.t


@dataclass
class LWEPublicKey:
    A: np.ndarray   # shape: (m, n)
    b: np.ndarray   # shape: (m,)
    params: LWEParams


@dataclass
class LWEPrivateKey:
    s: np.ndarray   # shape: (n,)
    params: LWEParams


@dataclass
class LWECiphertext:
    u: np.ndarray   # shape: (n,)
    v: int          # scalar
    t: int          # 本次加密的 bit 數


@dataclass
class AttackResult:
    success: bool
    elapsed: float
    recovered_s: Optional[np.ndarray]
    method: str
    attempts: int
    estimated_total: int = 0    # q^n，理論最大嘗試次數


@dataclass
class EncryptionStats:
    """UTF-8 加密的統計資訊"""
    text: str
    encoding: str
    byte_count: int
    bit_count: int
    t: int
    encrypt_count: int          # 需要幾次 LWE 加密
    encrypt_time: float         # 總加密時間（秒）
    decrypt_time: float         # 總解密時間（秒）
    error_count: int            # 解密錯誤的 bit 組數
    recovered_text: str         # 解密還原的文字


# ─────────────────────────────────────────────────────────────────────────────
# 輔助函數
# ─────────────────────────────────────────────────────────────────────────────

def is_prime(n: int) -> bool:
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


def validate_params(n: int, q: int, k: int, m: int, t: int = 1) -> str:
    """
    驗證 LWE 參數合法性
    回傳空字串表示合法，否則回傳錯誤訊息
    """
    if n < 2:
        return f"維度 n={n} 必須 ≥ 2（n=1 時只有 q 種可能，毫秒內可破，無密碼學意義）"
    if not is_prime(q):
        return f"模數 q={q} 必須是質數"
    if k < 1:
        return f"雜訊幅度 k={k} 必須 ≥ 1"
    if t < 1:
        return f"每次加密 bit 數 t={t} 必須 ≥ 1"
    if 2**t > q:
        return f"t={t} 太大：2^t={2**t} 超過 q={q}，無法分割成 2^t 個區間"
    if m < n:
        return f"公鑰行數 m={m} 建議 ≥ n={n}"
    rho = noise_ratio(k, q, t=t)
    if rho >= 1.0:
        return (f"警告：廣義雜訊比 ρ_t = k/(q/2^(t+1)) = {rho:.3f} ≥ 1.0，"
                f"解密將大量失敗。建議降低 k、增大 q 或減小 t。")
    if rho >= 0.5:
        return (f"警告：廣義雜訊比 ρ_t = {rho:.3f}，"
                f"接近相變點，解密可能出現錯誤。")
    return ""


def noise_ratio(k: int, q: int, t: int = 1) -> float:
    """
    廣義雜訊比 ρ_t = k / (q / 2^(t+1))

    t=1 時退化為原始定義 ρ = k / (q/4)
    ρ_t < 1  → 解密安全區
    ρ_t ≈ 1  → 相變點（解密開始大量失敗）
    ρ_t > 1  → 解密危險區
    """
    threshold = q / (2 ** (t + 1))
    return k / threshold if threshold > 0 else float('inf')


def encode_symbol(mu: int, q: int, t: int) -> int:
    """
    將符號 μ ∈ {0, …, 2^t - 1} 編碼為 ℤq 中的值

    將 q 均分成 2^t 個區間，μ 對應第 μ 個區間的中心點：
      encode(μ) = round(μ × q / 2^t)  mod q
    """
    symbols = 2 ** t
    return round(mu * q / symbols) % q


def decode_symbol(d: int, q: int, t: int) -> int:
    """
    從 ℤq 中的值 d 解碼回符號 μ ∈ {0, …, 2^t - 1}

    找到使 d 最接近 encode(μ) 的 μ（在模 q 的環上比較距離）
    """
    symbols = 2 ** t
    best_mu = 0
    best_dist = float('inf')
    for mu in range(symbols):
        center = encode_symbol(mu, q, t)
        dist = min(abs(d - center), q - abs(d - center))
        if dist < best_dist:
            best_dist = dist
            best_mu = mu
    return best_mu


def theoretical_fail_rate(k: int, q: int, m: int, t: int = 1) -> float:
    """
    理論解密失敗率（常態近似）

    推導：
      rᵀe 的分布近似 N(0, σ²)，其中 σ² = k² × w / 3，w = m/2（期望漢明重量）
      解密正確條件：|rᵀe| < q / 2^(t+1)
      P_fail ≈ 2 × Φ(-threshold / σ)
    """
    try:
        from scipy import stats
    except ImportError:
        return None

    w = m / 2
    sigma = np.sqrt(k**2 * w / 3)
    threshold = q / (2 ** (t + 1))
    if sigma == 0:
        return 0.0
    p_fail = 2 * stats.norm.cdf(-threshold / sigma)
    return float(np.clip(p_fail, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 金鑰生成
# ─────────────────────────────────────────────────────────────────────────────

def keygen(params: LWEParams,
           rng: Optional[np.random.Generator] = None
           ) -> tuple[LWEPublicKey, LWEPrivateKey]:
    """
    LWE 金鑰生成

    1. 隨機選取私鑰 s ∈ ℤqⁿ
    2. 隨機選取公開矩陣 A ∈ ℤqᵐˣⁿ（完全隨機）
    3. 從 U[-k, k] 採樣雜訊向量 e
    4. 計算公鑰 b = As + e (mod q)

    注意：A 使用完全隨機產生（非種子展開、非循環矩陣），
    安全性假設最少、數學分析最清楚，適合研究用途。
    """
    if rng is None:
        rng = np.random.default_rng()

    n, q, k, m = params.n, params.q, params.k, params.m
    s = rng.integers(0, q, size=n)
    A = rng.integers(0, q, size=(m, n))
    e = rng.integers(-k, k + 1, size=m)
    b = (A @ s + e) % q

    return LWEPublicKey(A=A, b=b, params=params), LWEPrivateKey(s=s, params=params)


# ─────────────────────────────────────────────────────────────────────────────
# 加密 / 解密（t bits）
# ─────────────────────────────────────────────────────────────────────────────

def encrypt(pub: LWEPublicKey, mu: int,
            rng: Optional[np.random.Generator] = None,
            min_weight_ratio: float = 0.25) -> LWECiphertext:
    """
    LWE 加密（t bits）

    加密符號 μ ∈ {0, 1, …, 2^t - 1}

    流程：
      1. 選取 r ∈ {0,1}ᵐ，漢明重量 ≥ m × min_weight_ratio
         （排除全零及過稀疏的 r，確保語意安全性）
      2. u = rᵀA  (mod q)
      3. v = rᵀb + encode(μ, q, t)  (mod q)
    """
    if rng is None:
        rng = np.random.default_rng()

    params = pub.params
    n, q, m, t = params.n, params.q, params.m, params.t
    symbols = 2 ** t

    if not (0 <= mu < symbols):
        raise ValueError(f"符號 μ={mu} 必須在 [0, {symbols-1}] 之間（t={t}）")

    min_weight = max(1, int(m * min_weight_ratio))
    encoded = encode_symbol(mu, q, t)

    for _ in range(10000):
        r = rng.integers(0, 2, size=m)
        if np.sum(r) >= min_weight:
            break
    else:
        raise RuntimeError("無法生成有效的 r，請增大 m 或降低 min_weight_ratio")

    u = (r @ pub.A) % q
    v = int((r @ pub.b + encoded) % q)
    return LWECiphertext(u=u, v=v, t=t)


def decrypt(priv: LWEPrivateKey, ct: LWECiphertext) -> int:
    """
    LWE 解密

    流程：
      1. d = v - u·s  (mod q)
         = rᵀe + encode(μ)  (mod q)
      2. 找 d 最接近的區間中心點，對應的 μ 即為解密結果

    正確性條件：|rᵀe| < q / 2^(t+1)
    """
    q = priv.params.q
    t = ct.t
    d = int((ct.v - int(ct.u @ priv.s)) % q)
    return decode_symbol(d, q, t)


# ─────────────────────────────────────────────────────────────────────────────
# 批次正確率測試
# ─────────────────────────────────────────────────────────────────────────────

def measure_error_rate(params: LWEParams,
                       trials: int = 1000,
                       seed: Optional[int] = None) -> dict:
    """
    執行 trials 次獨立加解密，統計解密錯誤率

    每次使用新的金鑰對，確保統計獨立性
    """
    rng = np.random.default_rng(seed)
    symbols = 2 ** params.t
    errors = 0

    for _ in range(trials):
        pub, priv = keygen(params, rng=rng)
        mu = int(rng.integers(0, symbols))
        ct = encrypt(pub, mu, rng=rng)
        mu_hat = decrypt(priv, ct)
        if mu_hat != mu:
            errors += 1

    theory = theoretical_fail_rate(params.k, params.q, params.m, params.t)

    return {
        "error_rate":  errors / trials,
        "error_count": errors,
        "trials":      trials,
        "rho":         params.rho,
        "rho_t":       params.rho_t,
        "theory_fail": theory,
        "n": params.n, "q": params.q, "k": params.k,
        "m": params.m, "t": params.t,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UTF-8 / ASCII / Big5 字串加解密
# ─────────────────────────────────────────────────────────────────────────────

def encrypt_text(pub: LWEPublicKey, text: str,
                 encoding: str = "utf-8",
                 rng: Optional[np.random.Generator] = None) -> list[LWECiphertext]:
    """
    加密任意字串（UTF-8 / ASCII / Big5）

    流程：
      1. 將字串用指定編碼轉換為 bytes
      2. 將 bytes 轉換為 bit 序列
      3. 按照 t bits 一組分組，不足補零
      4. 對每組 t bits 執行一次 LWE 加密

    回傳：密文列表（每個元素對應一組 t bits）
    """
    if rng is None:
        rng = np.random.default_rng()

    t = pub.params.t
    try:
        raw_bytes = text.encode(encoding)
    except (UnicodeEncodeError, LookupError) as e:
        raise ValueError(f"編碼失敗（{encoding}）：{e}")

    # 轉成 bit 序列
    bits = []
    for byte in raw_bytes:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    # 補齊到 t 的倍數
    while len(bits) % t != 0:
        bits.append(0)

    # 每 t bits 一組加密
    ciphertexts = []
    for i in range(0, len(bits), t):
        group = bits[i:i+t]
        mu = int("".join(str(b) for b in group), 2)
        ciphertexts.append(encrypt(pub, mu, rng=rng))

    return ciphertexts


def decrypt_text(priv: LWEPrivateKey,
                 ciphertexts: list[LWECiphertext],
                 encoding: str = "utf-8",
                 original_byte_count: Optional[int] = None) -> tuple[str, int]:
    """
    解密密文列表，還原字串

    回傳：(還原的字串, 解密錯誤的符號組數)
    """
    t = priv.params.t
    bits = []
    errors = 0

    for ct in ciphertexts:
        mu_hat = decrypt(priv, ct)
        group_bits = [(mu_hat >> (t - 1 - i)) & 1 for i in range(t)]
        bits.extend(group_bits)

    # 還原 bytes
    if original_byte_count is not None:
        bits = bits[:original_byte_count * 8]

    # 補齊到 8 的倍數
    while len(bits) % 8 != 0:
        bits.append(0)

    raw_bytes = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = int("".join(str(b) for b in bits[i:i+8]), 2)
        raw_bytes.append(byte_val)

    try:
        recovered = raw_bytes.decode(encoding, errors="replace")
    except Exception:
        recovered = ""

    return recovered, errors


def encrypt_text_timed(params: LWEParams, text: str,
                       encoding: str = "utf-8",
                       seed: Optional[int] = None) -> EncryptionStats:
    """
    加解密字串並計時，回傳完整統計資訊
    """
    rng = np.random.default_rng(seed)

    try:
        raw_bytes = text.encode(encoding)
    except (UnicodeEncodeError, LookupError) as e:
        raise ValueError(f"編碼失敗（{encoding}）：{e}")

    byte_count = len(raw_bytes)
    bit_count  = byte_count * 8
    t          = params.t
    enc_count  = -(-bit_count // t)   # ceiling division

    # 金鑰生成
    pub, priv = keygen(params, rng=rng)

    # 加密計時
    t0 = time.perf_counter()
    ciphertexts = encrypt_text(pub, text, encoding=encoding, rng=rng)
    encrypt_time = time.perf_counter() - t0

    # 解密計時
    t0 = time.perf_counter()
    recovered, error_count = decrypt_text(
        priv, ciphertexts, encoding=encoding,
        original_byte_count=byte_count
    )
    decrypt_time = time.perf_counter() - t0

    return EncryptionStats(
        text=text,
        encoding=encoding,
        byte_count=byte_count,
        bit_count=bit_count,
        t=t,
        encrypt_count=enc_count,
        encrypt_time=encrypt_time,
        decrypt_time=decrypt_time,
        error_count=error_count,
        recovered_text=recovered,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 暴力搜尋攻擊
# ─────────────────────────────────────────────────────────────────────────────

def brute_force_attack(pub: LWEPublicKey,
                       timeout: float = 60.0) -> AttackResult:
    """
    暴力搜尋攻擊：窮舉所有 s ∈ ℤqⁿ

    複雜度：O(q^n)

    判斷標準：候選 s_cand 使殘差 |b - As_cand| (mod q 中心化) 全部 ≤ k
    """
    from itertools import product

    params = pub.params
    n, q, k, m = params.n, params.q, params.k, params.m
    A, b = pub.A, pub.b
    estimated = q ** n

    start = time.time()
    attempts = 0

    for s_cand_tuple in product(range(q), repeat=n):
        if time.time() - start > timeout:
            return AttackResult(
                success=False, elapsed=time.time() - start,
                recovered_s=None, method="brute_force",
                attempts=attempts, estimated_total=estimated
            )
        attempts += 1
        s_cand = np.array(s_cand_tuple, dtype=np.int64)
        residuals = (b - A @ s_cand) % q
        residuals_c = np.where(residuals > q // 2, residuals - q, residuals)
        if np.all(np.abs(residuals_c) <= k):
            return AttackResult(
                success=True, elapsed=time.time() - start,
                recovered_s=s_cand, method="brute_force",
                attempts=attempts, estimated_total=estimated
            )

    return AttackResult(
        success=False, elapsed=time.time() - start,
        recovered_s=None, method="brute_force",
        attempts=attempts, estimated_total=estimated
    )


# ─────────────────────────────────────────────────────────────────────────────
# BKZ 攻擊介面（需安裝 fpylll）
# ─────────────────────────────────────────────────────────────────────────────

def bkz_attack(pub: LWEPublicKey,
               block_size: int = 10,
               timeout: float = 60.0) -> AttackResult:
    """
    BKZ 格基約化攻擊

    需要安裝 fpylll：pip install fpylll
    若未安裝，回傳 method="bkz_unavailable"
    """
    try:
        from fpylll import IntegerMatrix, BKZ, LLL
    except ImportError:
        return AttackResult(
            success=False, elapsed=0.0, recovered_s=None,
            method="bkz_unavailable", attempts=0,
            estimated_total=pub.params.q ** pub.params.n
        )

    params = pub.params
    n, q, k, m = params.n, params.q, params.k, params.m
    A, b = pub.A, pub.b
    start = time.time()

    dim = m + n + 1
    M = IntegerMatrix(m + 1, dim)
    for i in range(m):
        M[i, i] = q
    for i in range(m):
        for j in range(n):
            M[i, m + j] = int(A[i, j])
        M[m, i] = int(b[i])
    M[m, dim - 1] = 1

    try:
        LLL.reduction(M)
        BKZ.reduction(M, BKZ.Param(block_size=block_size))
    except Exception:
        return AttackResult(
            success=False, elapsed=time.time() - start,
            recovered_s=None, method="bkz_error", attempts=1,
            estimated_total=q ** n
        )

    for i in range(M.nrows):
        row = [M[i][j] for j in range(dim)]
        if row[dim - 1] in (1, -1):
            sign = row[dim - 1]
            s_cand = np.array(
                [sign * row[m + j] % q for j in range(n)], dtype=np.int64
            )
            residuals = (b - A @ s_cand) % q
            residuals_c = np.where(residuals > q // 2, residuals - q, residuals)
            if np.all(np.abs(residuals_c) <= k):
                return AttackResult(
                    success=True, elapsed=time.time() - start,
                    recovered_s=s_cand, method="bkz", attempts=1,
                    estimated_total=q ** n
                )

    return AttackResult(
        success=False, elapsed=time.time() - start,
        recovered_s=None, method="bkz", attempts=1,
        estimated_total=q ** n
    )
