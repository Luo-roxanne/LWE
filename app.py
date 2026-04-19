"""
app.py  ──  Streamlit 互動展示介面
─────────────────────────────────────────────────────────────────────────────
旺宏科學獎：格密碼 LWE 安全性分析
實驗一：維度基準線

功能：
  頁籤一  即時加解密示範（任意輸入 n, q, k）
  頁籤二  實驗一互動執行（掃描維度 n，即時繪圖）
  頁籤三  載入已存結果並視覺化

執行：
  streamlit run app.py
"""

import json
import os
import time
from typing import Optional

import numpy as np
import streamlit as st

# ── 頁面基本設定 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LWE 安全性分析｜旺宏科學獎",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 匯入核心模組（同目錄）────────────────────────────────────────────────────
try:
    from lwe_core import (
        keygen, encrypt, decrypt, brute_force_attack,
        measure_error_rate, noise_ratio, theoretical_fail_rate,
        validate_params, LWEPublicKey, LWEPrivateKey
    )
    CORE_OK = True
except ImportError as e:
    st.error(f"無法匯入 lwe_core.py：{e}")
    st.stop()

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# 側邊欄：全域參數
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔐 LWE 參數設定")
    st.markdown("---")

    st.subheader("核心參數")
    n_val = st.slider("維度 n", min_value=1, max_value=20, value=4,
                      help="秘密向量的維度，安全性的主要來源")
    q_val = st.selectbox("模數 q（質數）",
                         [17, 23, 31, 41, 53, 71, 97, 101, 127, 257],
                         index=7,
                         help="LWE 的模數，控制運算範圍（注意：q 越大安全性不一定越高，主要取決於 n）")
    k_val = st.slider("雜訊幅度 k",
                      min_value=1,
                      max_value=max(1, q_val // 4 + 3),
                      value=min(2, q_val // 8),
                      help="雜訊從 U[-k, k] 均勻分布採樣")
    m_val = 2 * n_val

    st.markdown("---")

    # 即時計算安全指標
    rho = noise_ratio(k_val, q_val)
    warn = validate_params(n_val, q_val, k_val, m_val)

    st.subheader("📊 安全性指標")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("雜訊比 ρ", f"{rho:.3f}",
                  help="ρ = k / (q/4)，ρ < 1 為安全區，ρ ≈ 1 為相變點")
    with col2:
        if rho < 0.5:
            st.metric("安全狀態", "✅ 安全")
        elif rho < 0.9:
            st.metric("安全狀態", "⚠️ 邊界")
        else:
            st.metric("安全狀態", "❌ 危險")

    try:
        theory = theoretical_fail_rate(k_val, q_val, m_val)
        st.metric("理論失敗率", f"{theory*100:.4f}%")
    except Exception:
        st.metric("理論失敗率", "需要 scipy")

    if warn:
        st.warning(warn)

    st.markdown("---")
    st.caption(
        "公鑰行數 m = 2n（自動設定）\n\n"
        "雜訊比 ρ = k / (q/4)\n"
        "- ρ << 1：解密幾乎不失敗\n"
        "- ρ ≈ 1 ：相變點\n"
        "- ρ >> 1：大量解密失敗"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主區域：三個頁籤
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔒 即時加解密示範",
    "🧪 實驗一：維度掃描",
    "📈 載入實驗結果",
])


# ══════════════════════════════════════════════════════════════════════════════
# 頁籤一：即時加解密示範
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("LWE 加解密完整流程示範")
    st.markdown(
        f"**目前參數：** n={n_val}, q={q_val}, k={k_val}, m={m_val}（m = 2n）"
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("① 金鑰生成")
        seed_input = st.number_input("隨機種子（可重現）", value=42, step=1)
        gen_btn = st.button("🔑 生成金鑰對", type="primary",
                            use_container_width=True)

        if gen_btn or "pub" not in st.session_state:
            rng = np.random.default_rng(int(seed_input))
            pub, priv = keygen(n_val, q_val, k_val, m_val, rng=rng)
            st.session_state["pub"] = pub
            st.session_state["priv"] = priv
            st.session_state["rng_seed"] = int(seed_input)

        pub  = st.session_state.get("pub")
        priv = st.session_state.get("priv")

        if pub and priv:
            st.success("金鑰生成成功！")

            with st.expander("🔍 查看私鑰 s（實際中絕不公開）"):
                st.code(f"s = {priv.s.tolist()}", language="python")

            with st.expander("🔍 查看公鑰 (A, b)"):
                st.markdown("**公開矩陣 A（前三行）：**")
                st.code(str(pub.A[:3].tolist()), language="python")
                st.markdown("**公鑰向量 b：**")
                st.code(str(pub.b.tolist()), language="python")
                st.caption("Eve 只能看到 A 和 b，無法從中還原 s（這正是 LWE 的困難性）")

    with col_right:
        st.subheader("② 加密")
        mu_input = st.radio("選擇要加密的訊息 μ", [0, 1], horizontal=True)

        enc_btn = st.button("🔒 加密", type="primary",
                            use_container_width=True)
        if enc_btn:
            if pub:
                rng2 = np.random.default_rng(
                    st.session_state.get("rng_seed", 42) + 1
                )
                ct = encrypt(pub, mu_input, rng=rng2)
                st.session_state["ct"] = ct
                st.session_state["mu_sent"] = mu_input
            else:
                st.error("請先生成金鑰")

        ct = st.session_state.get("ct")
        if ct:
            st.success(f"加密完成！明文 μ = {st.session_state.get('mu_sent')}")
            with st.expander("🔍 查看密文 (u, v)"):
                st.markdown("**密文向量 u：**")
                st.code(str(ct.u.tolist()), language="python")
                st.markdown(f"**密文純量 v = {ct.v}**")
                st.caption(
                    f"Eve 看到 (u, v) 但不知道 s，"
                    f"無法還原 μ（暴力破解需嘗試 q^n = {q_val}^{n_val} = "
                    f"{q_val**n_val:,} 種可能）"
                )

    # ── 解密 ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("③ 解密")

    if ct and priv:
        dec_btn = st.button("🔓 解密", type="primary", use_container_width=False)
        if dec_btn or "mu_recv" in st.session_state:
            mu_recv = decrypt(priv, ct)
            st.session_state["mu_recv"] = mu_recv

            mu_sent = st.session_state.get("mu_sent")
            correct = (mu_recv == mu_sent)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("原始明文 μ", str(mu_sent))
            with col_b:
                st.metric("解密結果 μ̂", str(mu_recv))
            with col_c:
                if correct:
                    st.metric("解密結果", "✅ 正確")
                else:
                    st.metric("解密結果", "❌ 錯誤")

            # 數學步驟展示
            with st.expander("📐 查看解密數學步驟"):
                q = priv.q
                d_raw = int(ct.v) - int(ct.u @ priv.s)
                d = d_raw % q
                half_q = q // 2
                dist0 = min(d, q - d)
                disth = min(abs(d - half_q), q - abs(d - half_q))
                noise_est = d - half_q if mu_sent == 1 else d

                st.code(
                    f"v - u·s (mod q)\n"
                    f"= {ct.v} - {int(ct.u @ priv.s) % q}  (mod {q})\n"
                    f"= {d}\n\n"
                    f"距離 0 的距離    = {dist0}\n"
                    f"距離 ⌊q/2⌋={half_q} 的距離 = {disth}\n\n"
                    f"{'→ 解碼為 1（更靠近 ⌊q/2⌋）' if disth < dist0 else '→ 解碼為 0（更靠近 0）'}\n"
                    f"實際雜訊項 rᵀe ≈ {noise_est}，閾值 ⌊q/4⌋ = {q//4}",
                    language="text"
                )
    else:
        st.info("請先完成金鑰生成與加密步驟")

    # ── 批次錯誤率測試 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("④ 批次解密錯誤率測試")
    st.caption(f"目前雜訊比 ρ = {rho:.3f}，{'解密應非常穩定' if rho < 0.5 else '接近相變點，可能出現錯誤'}")

    batch_trials = st.slider("測試次數", 100, 2000, 500, step=100)
    batch_btn = st.button("🔁 執行批次測試", use_container_width=False)

    if batch_btn:
        with st.spinner("測試中..."):
            result = measure_error_rate(
                n_val, q_val, k_val, m_val,
                trials=batch_trials, seed=42
            )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("解密錯誤率", f"{result['error_rate']*100:.2f}%")
        with col2:
            st.metric("錯誤次數", f"{result['error_count']}/{batch_trials}")
        with col3:
            if result["theory_fail"] is not None:
                st.metric("理論失敗率", f"{result['theory_fail']*100:.4f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 頁籤二：實驗一互動執行
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("實驗一：維度基準線")
    st.markdown(
        "固定 **q=101, k=2, m=2n**，將維度 n 從小掃到大，"
        "即時觀察破解時間與解密錯誤率的變化。"
    )

    col_cfg, col_run = st.columns([1, 2])

    with col_cfg:
        st.subheader("實驗設定")
        exp1_n_max   = st.slider("最大維度 n_max", 4, 20, 12)
        exp1_trials  = st.slider("每 n 的測試次數", 100, 1000, 300, step=100)
        exp1_timeout = st.slider("攻擊超時（秒）", 5, 60, 30)
        exp1_q       = 101
        exp1_k       = 2

        st.markdown(
            f"**固定參數：** q={exp1_q}, k={exp1_k}  \n"
            f"**雜訊比 ρ：** {noise_ratio(exp1_k, exp1_q):.3f}（固定）  \n"
            f"**總計算量：** 約 {exp1_n_max * exp1_trials:,} 次加解密"
        )
        run_exp1_btn = st.button("▶️ 開始實驗", type="primary",
                                 use_container_width=True)

    with col_run:
        if run_exp1_btn:
            results_live = []
            progress_bar = st.progress(0)
            status_text  = st.empty()

            chart_placeholder = st.empty()
            table_placeholder = st.empty()

            n_list = list(range(2, exp1_n_max + 1))

            for i, n in enumerate(n_list):
                m = 2 * n
                status_text.text(
                    f"執行中：n={n}，解密正確率測試（{exp1_trials} 次）..."
                )

                # 解密錯誤率
                err = measure_error_rate(
                    n, exp1_q, exp1_k, m,
                    trials=exp1_trials, seed=42 + i
                )

                # 攻擊（若 n 夠小）
                atk_elapsed = None
                atk_success = None
                if n <= 10:
                    status_text.text(
                        f"執行中：n={n}，暴力搜尋攻擊（超時 {exp1_timeout}s）..."
                    )
                    rng_atk = np.random.default_rng(42 + i)
                    pub_atk, _ = keygen(n, exp1_q, exp1_k, m, rng=rng_atk)
                    atk = brute_force_attack(pub_atk, timeout=exp1_timeout)
                    atk_elapsed = atk.elapsed
                    atk_success = atk.success

                results_live.append({
                    "n": n, "m": m,
                    "error_rate":    err["error_rate"],
                    "atk_elapsed":   atk_elapsed,
                    "atk_success":   atk_success,
                })

                # 即時更新圖表
                ns        = [r["n"] for r in results_live]
                err_rates = [r["error_rate"] * 100 for r in results_live]
                atk_times = [r["atk_elapsed"] for r in results_live
                             if r["atk_elapsed"] is not None]
                atk_ns    = [r["n"] for r in results_live
                             if r["atk_elapsed"] is not None]

                if PLOTLY_OK:
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=ns, y=err_rates,
                        name="解密錯誤率 (%)",
                        line=dict(color="#2E75B6", width=2),
                        marker=dict(size=8),
                        yaxis="y1",
                    ))

                    if atk_times:
                        fig.add_trace(go.Scatter(
                            x=atk_ns, y=atk_times,
                            name="攻擊時間 (秒)",
                            line=dict(color="#E05252", width=2,
                                      dash="dash"),
                            marker=dict(size=8,
                                        symbol=["circle" if s else "x"
                                                for s in
                                                [r["atk_success"]
                                                 for r in results_live
                                                 if r["atk_elapsed"] is not None]]),
                            yaxis="y2",
                        ))

                    fig.update_layout(
                        title=f"實驗一進度（n=2 到 {n}）",
                        xaxis_title="維度 n",
                        yaxis=dict(
                            title="解密錯誤率 (%)",
                            color="#2E75B6",
                            range=[-5, 105],
                        ),
                        yaxis2=dict(
                            title="攻擊時間（秒）",
                            color="#E05252",
                            overlaying="y",
                            side="right",
                            type="log",
                        ),
                        legend=dict(x=0.01, y=0.99),
                        height=400,
                    )
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                else:
                    chart_placeholder.line_chart(
                        {"解密錯誤率(%)": dict(zip(ns, err_rates))}
                    )

                progress_bar.progress((i + 1) / len(n_list))

            status_text.success("✅ 實驗一完成！")
            st.session_state["exp1_results"] = results_live

            # 摘要表格
            import pandas as pd
            df = pd.DataFrame(results_live)
            df["解密錯誤率"] = df["error_rate"].map(lambda x: f"{x*100:.2f}%")
            df["攻擊結果"] = df.apply(
                lambda row: (
                    "✓ 成功" if row["atk_success"] is True
                    else ("✗ 超時" if row["atk_success"] is False
                          else "（跳過）")
                ), axis=1
            )
            df["攻擊時間"] = df["atk_elapsed"].map(
                lambda x: f"{x:.3f}s" if x else "─"
            )
            table_placeholder.dataframe(
                df[["n", "m", "解密錯誤率", "攻擊結果", "攻擊時間"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("調整左側設定後，按「▶️ 開始實驗」")


# ══════════════════════════════════════════════════════════════════════════════
# 頁籤三：載入實驗結果
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("載入已存實驗結果")
    st.markdown(
        "執行 `python experiment1.py` 後，將 `results/exp1_results.json` "
        "上傳到這裡查看圖表。"
    )

    uploaded = st.file_uploader(
        "上傳 exp1_results.json", type="json"
    )

    # 也可以直接讀本地的 results/
    if os.path.exists("results/exp1_results.json") and not uploaded:
        st.info("偵測到本地 results/exp1_results.json，自動載入中...")
        with open("results/exp1_results.json", encoding="utf-8") as f:
            data = json.load(f)
    elif uploaded:
        data = json.load(uploaded)
    else:
        data = None

    if data:
        import pandas as pd
        df = pd.DataFrame(data)

        st.subheader("數據總覽")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if PLOTLY_OK:
            st.subheader("視覺化")

            # 圖一：解密錯誤率 vs n
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df["n"], y=df["error_rate"] * 100,
                mode="lines+markers",
                name="解密錯誤率",
                line=dict(color="#2E75B6", width=2),
                marker=dict(size=8),
            ))
            # 加上理論失敗率曲線（若有）
            if "theory_fail" in df.columns:
                theory_vals = df["theory_fail"].dropna()
                if len(theory_vals) > 0:
                    fig1.add_trace(go.Scatter(
                        x=df["n"][theory_vals.index],
                        y=theory_vals * 100,
                        mode="lines",
                        name="理論失敗率（常態近似）",
                        line=dict(color="#E05252", width=1.5, dash="dot"),
                    ))
            fig1.update_layout(
                title="解密錯誤率 vs 維度 n",
                xaxis_title="維度 n",
                yaxis_title="解密錯誤率 (%)",
                height=350,
            )
            st.plotly_chart(fig1, use_container_width=True)

            # 圖二：攻擊時間 vs n（log scale）
            has_atk = df["attack_elapsed"].notna()
            if has_atk.any():
                df_atk = df[has_atk]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df_atk["n"],
                    y=df_atk["attack_elapsed"],
                    mode="lines+markers",
                    name="攻擊時間",
                    line=dict(color="#E8A838", width=2),
                    marker=dict(
                        size=10,
                        symbol=[
                            "circle" if s else "x"
                            for s in df_atk["attack_success"]
                        ],
                        color=[
                            "#2E75B6" if s else "#E05252"
                            for s in df_atk["attack_success"]
                        ],
                    ),
                ))
                # 超時線
                timeout_val = df_atk["attack_elapsed"].max()
                fig2.add_hline(
                    y=timeout_val,
                    line_dash="dash", line_color="gray",
                    annotation_text="超時門檻",
                )
                fig2.update_layout(
                    title="攻擊時間 vs 維度 n（log scale）",
                    xaxis_title="維度 n",
                    yaxis_title="攻擊時間（秒）",
                    yaxis_type="log",
                    height=350,
                )
                fig2.update_traces(
                    marker=dict(size=10),
                    selector=dict(mode="lines+markers"),
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.caption(
                    "● 藍點 = 攻擊成功，✗ 紅叉 = 超時。"
                    "攻擊時間呈指數增長，對應 O(q^n) 的計算複雜度。"
                )
        else:
            st.warning("請安裝 plotly 以啟用互動圖表：pip install plotly")
            st.dataframe(df)
