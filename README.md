# LWE 安全性分析｜旺宏科學獎

格密碼 LWE 問題參數空間的安全性結構分析

## 檔案結構

```
lwe_experiment1/
├── lwe_core.py        # 核心模組：金鑰生成、加解密、攻擊
├── experiment1.py     # 實驗一：維度基準線（命令列執行）
├── app.py             # Streamlit 互動展示介面
├── requirements.txt   # 套件需求
└── results/           # 實驗結果（執行後自動產生）
    ├── exp1_results.json
    └── exp1_results.csv
```

## 安裝

```bash
pip install -r requirements.txt
```

## 執行方式

### 方式一：命令列執行實驗（產生數據）

```bash
# 快速模式（n=2~15，每 n 測 200 次，約 2-5 分鐘）
python experiment1.py

# 完整模式（n=2~25，每 n 測 1000 次，約 30-60 分鐘）
python experiment1.py --full

# 自訂參數
python experiment1.py --n_max 12 --trials 500
```

### 方式二：Streamlit 互動介面

```bash
streamlit run app.py
```

開啟後：
- **頁籤一**：即時加解密示範（可調整側邊欄參數）
- **頁籤二**：實驗一互動執行（即時繪圖）
- **頁籤三**：載入已存結果視覺化

## 實驗參數

| 參數 | 值 | 說明 |
|------|-----|------|
| q | 101 | 固定模數（質數，q/4 ≈ 25） |
| k | 2 | 固定雜訊幅度 |
| m | 2n | 公鑰行數（隨 n 自動調整） |
| n | 2 → 30 | 掃描維度 |
| ρ | ≈ 0.08 | 雜訊比（固定，遠低於相變點） |

## 數學符號

```
n  維度（安全性主要來源）
q  模數（質數，控制運算範圍）
k  雜訊幅度（均勻分布 U[-k, k]）
m  公鑰行數（建議 m = 2n）
s  私鑰向量 ∈ ℤq^n
A  公開矩陣 ∈ ℤq^(m×n)
e  雜訊向量
b  公鑰：b = As + e (mod q)
ρ  雜訊比 = k / (q/4)
```

## 關於 BKZ 攻擊

`lwe_core.py` 已包含 BKZ 攻擊介面，需安裝 `fpylll`：

```bash
# Linux（需 C++ 開發環境）
pip install fpylll

# macOS
brew install fplll && pip install fpylll
```

若未安裝，程式會自動跳過 BKZ 攻擊，暴力搜尋攻擊仍可正常執行。
