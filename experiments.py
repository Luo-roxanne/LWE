"""
experiments.py  ── 七個實驗的執行腳本
─────────────────────────────────────────────────────────────────────────────
執行方式：
  python experiments.py --exp 1            # 執行實驗一
  python experiments.py --exp 2
  python experiments.py --exp 3            # 較慢，掃描所有 (q,k) 組合
  python experiments.py --exp 4
  python experiments.py --exp 5
  python experiments.py --exp 6
  python experiments.py --exp 7
  python experiments.py --exp all          # 依序執行全部（需數小時）
  python experiments.py --exp 1 --quick   # 快速模式（減少測試次數）
"""

import argparse
import csv
import json
import os
import time
from typing import Optional

import numpy as np

from lwe_core import (
    LWEParams, keygen, encrypt, decrypt,
    measure_error_rate, brute_force_attack, bkz_attack,
    noise_ratio, theoretical_fail_rate,
    encrypt_text_timed
)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 暴力搜尋上限：q^n > 此值時跳過
MAX_BRUTE_FORCE = 500_000_000
ATTACK_TIMEOUT  = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# 共用工具
# ─────────────────────────────────────────────────────────────────────────────

def save(data: list[dict], name: str):
    """存 JSON 和 CSV"""
    with open(f"{OUTPUT_DIR}/{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    if data:
        with open(f"{OUTPUT_DIR}/{name}.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    print(f"  ✓ 已存至 {OUTPUT_DIR}/{name}.json / .csv")


def run_attack(params: LWEParams, seed: int) -> dict:
    """執行暴力搜尋攻擊，回傳結果 dict"""
    estimated = params.q ** params.n
    if estimated > MAX_BRUTE_FORCE:
        return {
            "attack_success": None,
            "attack_elapsed": None,
            "attack_attempts": None,
            "attack_method": "skipped",
            "estimated_total": estimated,
        }
    rng = np.random.default_rng(seed)
    pub, priv = keygen(params, rng=rng)
    atk = brute_force_attack(pub, timeout=ATTACK_TIMEOUT)
    return {
        "attack_success":  atk.success,
        "attack_elapsed":  atk.elapsed,
        "attack_attempts": atk.attempts,
        "attack_method":   atk.method,
        "estimated_total": str(estimated),  # 轉字串避免 JSON/arrow OverflowError
    }


def sep(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def subsep(label: str):
    print(f"\n  {'─'*54}")
    print(f"  {label}")
    print(f"  {'─'*54}")


# ─────────────────────────────────────────────────────────────────────────────
# 實驗一：維度基準線
# ─────────────────────────────────────────────────────────────────────────────

def exp1(quick: bool = False):
    """
    固定 q=101, k=2, t=1, m=2n，掃描 n 從 2 到 n_max
    觀察：破解時間隨 n 指數增長，解密錯誤率維持接近 0%
    """
    sep("實驗一：維度基準線")
    Q, K, T = 101, 2, 1
    n_max  = 10 if quick else 20
    trials = 200 if quick else 1000
    results = []

    for i, n in enumerate(range(2, n_max + 1)):
        m = 2 * n
        params = LWEParams(n=n, q=Q, k=K, m=m, t=T)
        subsep(f"n={n}, q={Q}, k={K}, m={m}  |  ρ={params.rho:.3f}")

        # 解密錯誤率
        print(f"  [1/2] 解密錯誤率測試（{trials} 次）...", end="", flush=True)
        err = measure_error_rate(params, trials=trials, seed=42 + i)
        print(f" 完成  錯誤率: {err['error_rate']*100:.2f}%"
              f"  理論: {(err['theory_fail'] or 0)*100:.4f}%")

        # 攻擊
        print(f"  [2/2] 暴力搜尋攻擊"
              f"（q^n={Q}^{n}={Q**n:,}，上限 {MAX_BRUTE_FORCE:,}）...",
              end="", flush=True)
        atk = run_attack(params, seed=42 + i)
        if atk["attack_method"] == "skipped":
            print(f" 跳過（q^n 超過上限）")
        elif atk["attack_success"]:
            print(f" ✓ 成功（{atk['attack_elapsed']:.4f}s，"
                  f"{atk['attack_attempts']:,} 次）")
        else:
            print(f" ✗ 超時（{atk['attack_elapsed']:.1f}s，"
                  f"{atk['attack_attempts']:,} 次）")

        results.append({**err, **atk})
        save(results, "exp1_dimension_baseline")

    _print_summary_1(results)
    return results


def _print_summary_1(results):
    print(f"\n  {'n':>4} {'m':>4} {'q^n':>15} {'錯誤率':>8} {'攻擊':>8} {'時間':>10}")
    print(f"  {'─'*4} {'─'*4} {'─'*15} {'─'*8} {'─'*8} {'─'*10}")
    for r in results:
        n   = r['n']
        est = f"{r.get('estimated_total', 0):,}"
        err = f"{r['error_rate']*100:.2f}%"
        if r['attack_method'] == 'skipped':
            atk, t = '跳過', '─'
        elif r['attack_success']:
            atk, t = '✓ 成功', f"{r['attack_elapsed']:.3f}s"
        else:
            atk, t = '✗ 超時', f">{ATTACK_TIMEOUT}s"
        print(f"  {n:>4} {2*n:>4} {est:>15} {err:>8} {atk:>8} {t:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# 實驗二：相變點定位
# ─────────────────────────────────────────────────────────────────────────────

def exp2(quick: bool = False):
    """
    固定 n=8, q=101, t=1, m=16，掃描 k 從 1 到 28
    觀察：解密錯誤率在 k ≈ q/4 ≈ 25 附近急劇上升
    同時比較理論預測曲線與實驗值
    """
    sep("實驗二：相變點精確定位")
    N, Q, T, M = 8, 101, 1, 16
    k_max  = 28
    trials = 300 if quick else 1000
    results = []

    print(f"  固定：n={N}, q={Q}, t={T}, m={M}")
    print(f"  掃描：k = 1 到 {k_max}（q/4 ≈ {Q//4}，理論相變點）")
    print(f"  每 k 測試 {trials} 次\n")

    for k in range(1, k_max + 1):
        params = LWEParams(n=N, q=Q, k=k, m=M, t=T)
        err = measure_error_rate(params, trials=trials, seed=42 + k)
        theory = err['theory_fail'] or 0.0
        rho = params.rho

        marker = " ← 相變點附近" if 0.8 <= rho <= 1.2 else ""
        print(f"  k={k:>2}  ρ={rho:.3f}  "
              f"錯誤率={err['error_rate']*100:>6.2f}%  "
              f"理論={theory*100:>6.3f}%{marker}")
        results.append(err)
        save(results, "exp2_phase_transition")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 實驗三：安全性地圖（二維熱圖）
# ─────────────────────────────────────────────────────────────────────────────

def exp3(quick: bool = False):
    """
    固定 n=8, t=1, m=16，掃描所有 (q, k) 組合
    產出二維熱圖數據：橫軸 q，縱軸 k，顏色為解密錯誤率
    驗證安全邊界是否為 ρ = k/(q/4) ≈ 1 的等值線
    """
    sep("實驗三：安全性地圖（二維熱圖）")
    N, T, M = 8, 1, 16
    trials = 200 if quick else 500

    # 質數列表（17 到 127）
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    q_list = [q for q in sieve(127) if q >= 17]
    results = []
    total = sum(q // 4 + 5 for q in q_list)
    done = 0

    print(f"  固定：n={N}, t={T}, m={M}")
    print(f"  掃描：{len(q_list)} 個質數 × 各自的 k 範圍")
    print(f"  每組測試 {trials} 次，共約 {total} 組\n")

    for q in q_list:
        k_max = q // 4 + 5
        for k in range(1, k_max + 1):
            params = LWEParams(n=N, q=q, k=k, m=M, t=T)
            err = measure_error_rate(params, trials=trials,
                                     seed=42 + q * 100 + k)
            results.append(err)
            done += 1
            if done % 50 == 0:
                print(f"  進度：{done}/{total}  "
                      f"q={q}, k={k}, 錯誤率={err['error_rate']*100:.1f}%")

    save(results, "exp3_security_map")
    print(f"\n  ✓ 共 {len(results)} 組數據")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 實驗四：公鑰行數 m 的影響
# ─────────────────────────────────────────────────────────────────────────────

def exp4(quick: bool = False):
    """
    固定 n=8, q=101, k=2, t=1，掃描 m 從 n 到 5n
    觀察：
      - 解密錯誤率理論上不隨 m 改變（ρ_t 與 m 無關）
      - 攻擊難度隨 m 增加而降低（攻擊者獲得更多線性方程）
    """
    sep("實驗四：公鑰行數 m 的影響")
    N, Q, K, T = 8, 101, 2, 1
    trials = 300 if quick else 1000
    results = []

    print(f"  固定：n={N}, q={Q}, k={K}, t={T}")
    print(f"  掃描：m = {N} 到 {5*N}（建議值 m=2n={2*N}）\n")

    for m in range(N, 5 * N + 1):
        params = LWEParams(n=N, q=Q, k=K, m=m, t=T)
        err = measure_error_rate(params, trials=trials, seed=42 + m)
        marker = " ← 建議值" if m == 2 * N else ""
        print(f"  m={m:>3}  m/n={m/N:.1f}  "
              f"錯誤率={err['error_rate']*100:.2f}%{marker}")
        results.append(err)
        save(results, "exp4_m_effect")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 實驗五：縮小版 Kyber 參數搜尋
# ─────────────────────────────────────────────────────────────────────────────

def exp5(quick: bool = False):
    """
    固定 n=16, q=257（費馬質數 2^8+1），掃描 k
    目標：找到同時滿足以下條件的最大 k：
      1. 解密錯誤率 < 0.1%
      2. 暴力搜尋攻擊 60 秒內無法成功
    對比 Kyber-512（n=256, q=3329）的對應雜訊比
    """
    sep("實驗五：縮小版 Kyber 參數搜尋")
    N, Q, T = 16, 257, 1
    M = 2 * N
    trials = 500 if quick else 2000
    results = []

    print(f"  固定：n={N}, q={Q}（費馬質數 2⁸+1），t={T}, m={M}")
    print(f"  目標：錯誤率 < 0.1%，且攻擊困難")
    print(f"  對比：Kyber-512（n=256, q=3329, η=3）\n")

    best_k = None
    for k in range(1, Q // 4 + 1):
        params = LWEParams(n=N, q=Q, k=k, m=M, t=T)
        err = measure_error_rate(params, trials=trials, seed=42 + k)
        error_rate = err['error_rate']
        rho = params.rho

        # Kyber-512 的對應 ρ 估算：η=3，q=3329
        # σ ≈ 1.0，ρ_kyber ≈ η / (q/4) ≈ 3/832 ≈ 0.0036（極小）
        # 但 Kyber 的 n=256 提供足夠安全性

        status = ""
        if error_rate < 0.001:
            status = " ✓ 錯誤率達標"
            best_k = k

        print(f"  k={k:>3}  ρ={rho:.4f}  "
              f"錯誤率={error_rate*100:.3f}%{status}")

        results.append({**err, "target_met": error_rate < 0.001})
        save(results, "exp5_mini_kyber")

        if error_rate >= 0.01:   # 超過 1%，停止搜尋
            break

    if best_k:
        params_best = LWEParams(n=N, q=Q, k=best_k, m=M, t=T)
        print(f"\n  最佳 k = {best_k}，對應 ρ = {params_best.rho:.4f}")
        print(f"  Kyber-512 對應 ρ ≈ 0.004（n=256 提供主要安全性）")
        print(f"  結論：縮小版 n=16 的 ρ 仍合理，定性驗證參數設計邏輯")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 實驗六：每次加密 bit 數 t 的影響
# ─────────────────────────────────────────────────────────────────────────────

def exp6(quick: bool = False):
    """
    固定 n=8, q=101, k=2, m=16，掃描 t 從 1 到 5
    觀察：
      - 廣義雜訊比 ρ_t 隨 t 增大而上升
      - 解密錯誤率從 0% 逐步上升
      - 找出錯誤率 < 0.1% 條件下 t 的最大安全值
    核心結論：增大 t 是零和遊戲（效率↑，安全性↓）
              增大 n 才是兩全其美（效率↑，安全性↑）
    """
    sep("實驗六：每次加密 bit 數 t 的影響")
    N, Q, K, M = 8, 101, 2, 16
    t_max  = 5
    trials = 300 if quick else 1000
    results = []

    print(f"  固定：n={N}, q={Q}, k={K}, m={M}")
    print(f"  掃描：t = 1 到 {t_max}")
    print(f"  每 t 測試 {trials} 次\n")
    print(f"  {'t':>3} {'符號數':>6} {'區間寬度':>8} {'ρ_t':>8} "
          f"{'錯誤率':>8} {'理論':>8} {'加密次數(24bits)':>16}")
    print(f"  {'─'*3} {'─'*6} {'─'*8} {'─'*8} "
          f"{'─'*8} {'─'*8} {'─'*16}")

    safe_t_max = 1
    for t in range(1, t_max + 1):
        if 2**t > Q:
            print(f"  t={t}: 2^t={2**t} > q={Q}，跳過（無法分割區間）")
            break
        params = LWEParams(n=N, q=Q, k=K, m=M, t=t)
        err = measure_error_rate(params, trials=trials, seed=42 + t)

        symbols     = 2 ** t
        interval_w  = Q / symbols
        rho_t       = params.rho_t
        error_rate  = err['error_rate']
        theory      = err['theory_fail'] or 0.0
        enc_count   = -(-24 // t)   # 加密一個中文字（24 bits）所需次數

        if error_rate < 0.001:
            safe_t_max = t
            flag = " ✓"
        else:
            flag = " ✗ 超標"

        print(f"  {t:>3} {symbols:>6} {interval_w:>8.2f} {rho_t:>8.3f} "
              f"{error_rate*100:>7.2f}% {theory*100:>7.3f}%"
              f" {enc_count:>8} 次{flag}")

        results.append({
            **err,
            "symbols":    symbols,
            "interval_w": interval_w,
            "enc_count_24bits": enc_count,
            "safe": error_rate < 0.001,
        })
        save(results, "exp6_t_effect")

    print(f"\n  結論：在 n={N}, q={Q}, k={K} 下，t 的最大安全值 = {safe_t_max}")
    print(f"  對比：t=1（效率基準）vs t={safe_t_max}（最大安全 t）")
    if safe_t_max > 1:
        speedup = 24 // 1 / (-(-24 // safe_t_max))
        print(f"  加密中文字速度提升：{speedup:.1f} 倍（加密次數從 24 降至 {-(-24 // safe_t_max)}）")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 實驗七：UTF-8 中文訊息端對端加密效率
# ─────────────────────────────────────────────────────────────────────────────

def exp7(quick: bool = False):
    """
    測試 UTF-8 中文訊息在不同 (n, q, k, t) 下的加密效率與安全性
    觀察：加密時間、解密時間、加密次數、解密正確性
    找出「安全性與效率的最佳平衡點」
    """
    sep("實驗七：UTF-8 中文訊息端對端加密效率")

    # 測試文字：涵蓋中文、英文、數字、符號
    texts = [
        ("單一中文字", "你"),
        ("短句", "你好世界"),
        ("含英文", "Hello 世界 2024"),
        ("完整訊息", "量子電腦威脅密碼學安全"),
    ]

    # 測試的參數組合
    param_configs = [
        # (n,  q,   k, m,  t,  標籤)
        (4,   101,  2,  8,  1, "低維 t=1"),
        (4,   101,  2,  8,  2, "低維 t=2"),
        (8,   101,  2, 16,  1, "中維 t=1（基準）"),
        (8,   101,  2, 16,  2, "中維 t=2"),
        (8,   257,  3, 16,  1, "中維 q=257 t=1"),
        (16,  257,  3, 32,  1, "高維 t=1（迷你Kyber）"),
    ]

    results = []
    encoding = "utf-8"

    for text_label, text in texts:
        print(f"\n  文字：「{text}」（{len(text.encode(encoding))} bytes）")
        print(f"  {'參數':^20} {'加密次數':>8} {'加密時間':>10} "
              f"{'解密時間':>10} {'還原正確':>8}")
        print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")

        for n, q, k, m, t, label in param_configs:
            try:
                params = LWEParams(n=n, q=q, k=k, m=m, t=t)
            except ValueError:
                continue

            # 重複測試取平均（quick 模式只測一次）
            repeat = 1 if quick else 3
            enc_times, dec_times = [], []
            correct = True

            for rep in range(repeat):
                stats = encrypt_text_timed(
                    params, text, encoding=encoding, seed=42 + rep
                )
                enc_times.append(stats.encrypt_time)
                dec_times.append(stats.decrypt_time)
                if stats.recovered_text != text:
                    correct = False

            avg_enc = sum(enc_times) / len(enc_times)
            avg_dec = sum(dec_times) / len(dec_times)
            correct_str = "✓" if correct else "✗"

            print(f"  {label:^20} {stats.encrypt_count:>8} "
                  f"{avg_enc*1000:>8.2f}ms {avg_dec*1000:>8.2f}ms "
                  f"{correct_str:>8}")

            results.append({
                "text_label":    text_label,
                "text":          text,
                "encoding":      encoding,
                "byte_count":    stats.byte_count,
                "bit_count":     stats.bit_count,
                "n": n, "q": q, "k": k, "m": m, "t": t,
                "label":         label,
                "rho":           params.rho,
                "rho_t":         params.rho_t,
                "encrypt_count": stats.encrypt_count,
                "encrypt_ms":    avg_enc * 1000,
                "decrypt_ms":    avg_dec * 1000,
                "correct":       correct,
            })

    save(results, "exp7_utf8_efficiency")

    # 額外：比較三種編碼對同一中文字的 byte 數
    print(f"\n  {'─'*54}")
    print(f"  編碼比較：「你好」的 byte 數")
    print(f"  {'─'*54}")
    sample = "你好"
    for enc in ["utf-8", "big5", "ascii"]:
        try:
            b = sample.encode(enc)
            print(f"  {enc:>8}：{len(b)} bytes = {len(b)*8} bits"
                  f"（t=1 需 {len(b)*8} 次 LWE 加密）")
        except (UnicodeEncodeError, LookupError):
            print(f"  {enc:>8}：無法編碼「{sample}」")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

EXP_MAP = {
    "1": exp1, "2": exp2, "3": exp3, "4": exp4,
    "5": exp5, "6": exp6, "7": exp7,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="旺宏科學獎 LWE 安全性分析 ── 七個實驗"
    )
    parser.add_argument(
        "--exp", type=str, default="1",
        help="要執行的實驗編號（1~7），或 all 執行全部"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速模式（減少測試次數，適合除錯）"
    )
    args = parser.parse_args()

    total_start = time.time()

    if args.exp == "all":
        for key, fn in EXP_MAP.items():
            fn(quick=args.quick)
    elif args.exp in EXP_MAP:
        EXP_MAP[args.exp](quick=args.quick)
    else:
        print(f"未知的實驗編號：{args.exp}，請輸入 1~7 或 all")

    print(f"\n  總耗時：{time.time() - total_start:.1f}s")
    print(f"  所有結果已存至 {OUTPUT_DIR}/")
