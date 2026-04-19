"""
experiment1.py
─────────────────────────────────────────────────────────────────────────────
實驗一：維度基準線

研究問題：
  固定 q=101, k=2, m=2n，將維度 n 從 2 掃到 30，
  測量暴力搜尋攻擊的執行時間與解密錯誤率。

預期結果：
  - 破解時間隨 n 呈指數增長（O(q^n) 複雜度）
  - 解密錯誤率接近 0%（因 ρ = k/(q/4) ≈ 0.08，遠低於相變點）

執行方式：
  python experiment1.py             # 快速模式（n=2~15，每 n 測 200 次）
  python experiment1.py --full      # 完整模式（n=2~25，每 n 測 1000 次）
  python experiment1.py --n_max 12  # 自訂最大維度
"""

import argparse
import csv
import json
import os
import time
from typing import Optional

import numpy as np

from lwe_core import (
    keygen, encrypt, decrypt, brute_force_attack,
    measure_error_rate, noise_ratio, validate_params
)

# ─────────────────────────────────────────────────────────────────────────────
# 實驗參數
# ─────────────────────────────────────────────────────────────────────────────

Q  = 101    # 固定模數（質數，q/4 ≈ 25，教學用標準質數）
K  = 2      # 固定雜訊幅度（ρ = k/(q/4) ≈ 0.08，遠低於相變點）
ATTACK_TIMEOUT = 60.0   # 攻擊超時（秒）


# ─────────────────────────────────────────────────────────────────────────────
# 單一維度實驗
# ─────────────────────────────────────────────────────────────────────────────

def run_single_n(n: int, trials: int, seed: Optional[int] = None,
                 do_attack: bool = True, verbose: bool = True) -> dict:
    """
    對單一維度 n 執行完整實驗：
      1. 解密錯誤率測試（trials 次）
      2. 暴力搜尋攻擊計時（若 n ≤ 12）
    """
    m = 2 * n   # 公鑰行數：m = 2n

    if verbose:
        rho = noise_ratio(K, Q)
        print(f"\n{'─'*60}")
        print(f"  n={n}, q={Q}, k={K}, m={m}")
        print(f"  雜訊比 ρ = k/(q/4) = {rho:.3f}")
        print(f"{'─'*60}")

    # ── 1. 解密錯誤率 ──────────────────────────────────────────────
    if verbose:
        print(f"  [1/2] 解密正確率測試（{trials} 次）...", end="", flush=True)
    t0 = time.time()
    err_result = measure_error_rate(n, Q, K, m, trials=trials, seed=seed)
    t_err = time.time() - t0

    if verbose:
        print(f" 完成（{t_err:.2f}s）")
        print(f"       錯誤率: {err_result['error_rate']*100:.2f}%"
              f"  ({err_result['error_count']}/{trials})")
        if err_result["theory_fail"] is not None:
            print(f"       理論失敗率: {err_result['theory_fail']*100:.4f}%")

    # ── 2. 攻擊計時 ────────────────────────────────────────────────
    attack_result = None
    if do_attack and n <= 12:
        if verbose:
            print(f"  [2/2] 暴力搜尋攻擊（超時 {ATTACK_TIMEOUT}s）...",
                  end="", flush=True)
        rng = np.random.default_rng(seed)
        pub, priv = keygen(n, Q, K, m, rng=rng)
        attack_result = brute_force_attack(pub, timeout=ATTACK_TIMEOUT)

        if verbose:
            if attack_result.success:
                print(f" ✓ 成功（{attack_result.elapsed:.4f}s，"
                      f"{attack_result.attempts:,} 次嘗試）")
                # 驗證還原的 s 是否正確
                s_match = np.array_equal(attack_result.recovered_s, priv.s)
                if not s_match:
                    print(f"       ⚠ 警告：還原的 s 與真實 s 不符"
                          f"（可能有多解）")
            else:
                print(f" ✗ 超時（{attack_result.elapsed:.1f}s，"
                      f"{attack_result.attempts:,} 次嘗試）")
    elif do_attack and n > 12:
        if verbose:
            print(f"  [2/2] n={n} > 12，跳過暴力搜尋攻擊"
                  f"（O(q^n) = O({Q}^{n}) 太大）")

    return {
        "n": n,
        "q": Q,
        "k": K,
        "m": m,
        "rho": noise_ratio(K, Q),
        "error_rate": err_result["error_rate"],
        "error_count": err_result["error_count"],
        "trials": trials,
        "theory_fail": err_result["theory_fail"],
        "attack_success": attack_result.success if attack_result else None,
        "attack_elapsed": attack_result.elapsed if attack_result else None,
        "attack_attempts": attack_result.attempts if attack_result else None,
        "attack_method": attack_result.method if attack_result else "skipped",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主實驗迴圈
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment1(n_range: range, trials: int,
                    output_dir: str = "results",
                    seed: int = 42,
                    verbose: bool = True) -> list[dict]:
    """
    對每個 n 執行實驗並收集結果
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print("=" * 60)
    print("  實驗一：維度基準線")
    print(f"  q={Q}, k={K}, m=2n")
    print(f"  n 範圍：{n_range.start} 到 {n_range.stop - 1}")
    print(f"  每 n 測試次數：{trials}")
    print(f"  雜訊比 ρ = {K/(Q/4):.3f}（固定值）")
    print("=" * 60)

    total_start = time.time()

    for i, n in enumerate(n_range):
        result = run_single_n(
            n=n,
            trials=trials,
            seed=seed + i,
            do_attack=True,
            verbose=verbose,
        )
        results.append(result)

        # 即時存檔（防止中途中斷遺失數據）
        _save_results(results, output_dir)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  實驗一完成！總耗時：{total_elapsed:.1f}s")
    print(f"  結果已存至：{output_dir}/")
    print("=" * 60)

    # 印出摘要表格
    _print_summary(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 輸出函數
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(results: list[dict], output_dir: str):
    """存 JSON 和 CSV"""
    # JSON
    with open(os.path.join(output_dir, "exp1_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # CSV
    if results:
        keys = results[0].keys()
        with open(os.path.join(output_dir, "exp1_results.csv"), "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)


def _print_summary(results: list[dict]):
    """印出格式化摘要表格"""
    print("\n  📊 摘要表格")
    print(f"  {'n':>4} {'m':>4} {'錯誤率':>8} {'攻擊結果':>12} {'攻擊時間':>10}")
    print(f"  {'─'*4} {'─'*4} {'─'*8} {'─'*12} {'─'*10}")

    for r in results:
        n = r["n"]
        m = r["m"]
        err = f"{r['error_rate']*100:.2f}%"

        if r["attack_method"] == "skipped":
            atk_str = "（跳過）"
            time_str = "─"
        elif r["attack_success"]:
            atk_str = "✓ 成功"
            time_str = f"{r['attack_elapsed']:.3f}s"
        else:
            atk_str = "✗ 超時"
            time_str = f">{ATTACK_TIMEOUT}s"

        print(f"  {n:>4} {m:>4} {err:>8} {atk_str:>12} {time_str:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口點
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="旺宏科學獎 - 實驗一：維度基準線"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="完整模式：n=2~25，每 n 測 1000 次（較慢）"
    )
    parser.add_argument(
        "--n_max", type=int, default=None,
        help="自訂最大維度（覆蓋 --full 設定）"
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="自訂每 n 的測試次數"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="隨機種子（預設 42）"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="結果輸出目錄（預設 results/）"
    )
    args = parser.parse_args()

    # 決定參數
    if args.full:
        n_max   = args.n_max or 25
        trials  = args.trials or 1000
    else:
        n_max   = args.n_max or 15
        trials  = args.trials or 200

    run_experiment1(
        n_range=range(2, n_max + 1),
        trials=trials,
        output_dir=args.output,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
