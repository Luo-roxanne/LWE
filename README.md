# 格密碼 LWE 問題中雜訊分布對安全性邊界的影響

旺宏科學獎　數學組

## 研究主題

比較三種雜訊分布（高斯、均勻、CBD）對 LWE 安全性地圖的影響。

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `lwe_core_v2.py` | 核心模組，支援三種雜訊分布的採樣、加解密、錯誤率測試 |
| `lwe_three_dist.py` | 三個主實驗（A/B/C）的執行腳本 |
| `app_v2.py` | Streamlit 互動展示 App |
| `requirements.txt` | Python 套件需求 |

## 執行方式

### 安裝套件
```bash
pip install -r requirements.txt
```

### 執行實驗
```bash
python lwe_three_dist.py --exp A          # 實驗A：相變點比較
python lwe_three_dist.py --exp B          # 實驗B：安全性地圖（最慢）
python lwe_three_dist.py --exp C          # 實驗C：縮小版Kyber
python lwe_three_dist.py --exp all        # 全部執行
python lwe_three_dist.py --exp all --quick  # 快速模式
```

結果存於 `results3/` 資料夾。

### 開啟互動 App
```bash
streamlit run app_v2.py
```

## 三種雜訊分布說明

| 分布 | 參數 | rᵀe 標準差 | 安全性歸約 | 採樣效率 |
|---|---|---|---|---|
| 均勻 U(-k,k) | k | k√(m/6) | 較弱 | 中 |
| 高斯 DG(σ) | σ | σ√(m/2) | 最強 | 低 |
| CBD(η) | η | √(mη/4) | 中等 | 最高 |

有效雜訊比：`ρ_eff = σ_r / (q/4)`

安全邊界（n=8, m=16, q=17~127）：
- 均勻：ρ_eff ≤ 0.207（最嚴格）
- 高斯：ρ_eff ≤ 0.256
- CBD ：ρ_eff ≤ 0.320（最寬鬆）

## 參考文獻

- Regev, O. (2009). On lattices, learning with errors...
- NIST FIPS 203 (2024). Module-Lattice-Based KEM Standard.
- Bos et al. (2018). CRYSTALS-Kyber.
