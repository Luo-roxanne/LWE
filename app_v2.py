"""
app_v2.py  ── Streamlit App（三種雜訊分布版）
旺宏科學獎：格密碼 LWE 問題中雜訊分布對安全性邊界的影響

執行：streamlit run app_v2.py
"""
import streamlit as st
import numpy as np
import json
import os

st.set_page_config(
    page_title="LWE 三種雜訊分布安全性比較",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from lwe_core_v2 import (
        rho_eff, theory_pfail, measure_error_rate,
        encrypt_text, sample_uniform, sample_gaussian, sample_cbd,
    )
except ImportError as e:
    st.error(f"無法匯入 lwe_core_v2.py：{e}")
    st.stop()

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

# ── 側邊欄 ──────────────────────────────────────────────
with st.sidebar:
    st.title("🔐 LWE 參數設定")
    st.markdown("---")

    dist_choice = st.selectbox(
        "雜訊分布",
        ["uniform", "gaussian", "cbd"],
        format_func=lambda x: {
            "uniform": "均勻 U(-k, k)",
            "gaussian": "離散高斯 DG(σ)",
            "cbd": "中心二項 CBD(η)",
        }[x],
        help="三種分布的安全性歸約強度：高斯 ≥ CBD > 均勻"
    )

    n_val = st.slider("維度 n", 2, 20, 8)
    q_val = st.selectbox("模數 q（質數）",
        [17, 23, 31, 41, 53, 71, 97, 101, 127, 257], index=7)

    if dist_choice == "uniform":
        param_val = st.slider("雜訊幅度 k", 1, max(1, q_val // 4 + 3), 2)
        param_label = f"k = {param_val}"
    elif dist_choice == "gaussian":
        param_val = st.slider("標準差 σ", 0.5, float(q_val // 8 + 2), 1.5, step=0.5)
        param_label = f"σ = {param_val}"
    else:
        param_val = st.slider("參數 η（Kyber 用 η=3）", 1, 20, 3)
        param_label = f"η = {param_val}"

    m_val = 2 * n_val

    st.markdown("---")
    st.subheader("📊 安全性指標")

    reff = rho_eff(dist_choice, param_val, q_val, m_val)
    pfail = theory_pfail(reff)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("有效雜訊比 ρ_eff", f"{reff:.4f}")
    with col2:
        st.metric("理論失敗率", f"{pfail*100:.4f}%")

    if reff < 0.207:
        st.success(f"✅ 安全區（三種分布均安全）")
    elif reff < 0.256:
        st.warning(f"⚠️ 均勻分布不安全，高斯/CBD 安全")
    elif reff < 0.320:
        st.warning(f"⚠️ 均勻/高斯不安全，CBD 安全")
    else:
        st.error(f"❌ 三種分布均不安全")

    st.caption(
        f"m = 2n = {m_val}（自動）\n\n"
        f"安全邊界（均值）：\n"
        f"• 均勻：ρ_eff ≤ 0.207\n"
        f"• 高斯：ρ_eff ≤ 0.256\n"
        f"• CBD：ρ_eff ≤ 0.320"
    )

# ── 頁籤 ────────────────────────────────────────────────
tabs = st.tabs([
    "🔒 加解密示範",
    "📈 相變點比較",
    "🗺 安全性地圖",
    "🔬 縮小版Kyber",
    "🌏 UTF-8 效率",
    "📂 載入結果",
])
tab_demo, tab_phase, tab_map, tab_kyber, tab_utf8, tab_load = tabs

# ══════════════════════════════════════════════════════════
# 頁籤一：加解密示範
# ══════════════════════════════════════════════════════════
with tab_demo:
    st.header("LWE 加解密示範（三種雜訊分布）")
    st.caption(f"分布：{dist_choice}，{param_label}，n={n_val}，q={q_val}，m={m_val}")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("① 批次錯誤率測試")
        trials = st.slider("測試次數", 100, 2000, 500, step=100, key="demo_trials")
        seed = st.number_input("隨機種子", value=42, step=1, key="demo_seed")

        if st.button("▶️ 執行測試", type="primary", use_container_width=True):
            with st.spinner("測試中..."):
                result = measure_error_rate(
                    n_val, q_val, m_val, dist_choice, param_val,
                    trials=trials, seed=int(seed)
                )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("解密錯誤率", f"{result['error_rate']*100:.2f}%")
            with c2:
                st.metric("理論失敗率", f"{result['theory_pfail']*100:.4f}%")
            with c3:
                st.metric("有效雜訊比", f"{result['rho_eff']:.4f}")

            err_pct = result['error_rate'] * 100
            thy_pct = (result['theory_pfail'] or 0) * 100
            if err_pct < 0.1:
                st.success(f"✅ 安全：{trials} 次測試中 {result['error_count']} 次失敗（{err_pct:.2f}%）")
            elif err_pct < 5:
                st.warning(f"⚠️ 邊界：{trials} 次測試中 {result['error_count']} 次失敗（{err_pct:.2f}%）")
            else:
                st.error(f"❌ 不安全：{trials} 次測試中 {result['error_count']} 次失敗（{err_pct:.2f}%）")

            with st.expander("數學說明"):
                st.code(
                    f"分布：{dist_choice}，{param_label}\n"
                    f"rᵀe 的標準差 σ_r：\n"
                    f"  均勻：k√(m/6) = {param_val}×{np.sqrt(m_val/6):.3f}\n"
                    f"  高斯：σ_G√(m/2) = σ_G×{np.sqrt(m_val/2):.3f}\n"
                    f"  CBD ：√(mη/4) = √({m_val}η/4)\n"
                    f"有效雜訊比 ρ_eff = σ_r / (q/4) = {reff:.4f}\n"
                    f"理論失敗率 P_fail = 2Φ(-1/ρ_eff) = {thy_pct:.4f}%"
                )

    with col_r:
        st.subheader("② 三種分布快速對比")
        if st.button("📊 同時比較三種分布", use_container_width=True):
            results_3 = {}
            for d, p, lbl in [
                ("uniform", max(1, round(reff * (q_val/4) / np.sqrt(m_val/6))), "均勻"),
                ("gaussian", max(0.5, reff * (q_val/4) / np.sqrt(m_val/2)), "高斯"),
                ("cbd", max(1, round(4 * (reff * q_val/4)**2 / m_val)), "CBD"),
            ]:
                try:
                    r = measure_error_rate(n_val, q_val, m_val, d, p, trials=300, seed=42)
                    results_3[lbl] = r
                except Exception:
                    pass

            if results_3:
                st.markdown(f"**在 ρ_eff ≈ {reff:.3f} 下的比較：**")
                data = {
                    "分布": list(results_3.keys()),
                    "實驗錯誤率(%)": [v['error_rate']*100 for v in results_3.values()],
                    "理論失敗率(%)": [(v['theory_pfail'] or 0)*100 for v in results_3.values()],
                    "實際 ρ_eff": [v['rho_eff'] for v in results_3.values()],
                }
                import pandas as pd
                st.dataframe(pd.DataFrame(data).round(4),
                             hide_index=True, use_container_width=True)
                st.caption("相同 ρ_eff 下，均勻分布錯誤率通常最高，CBD 和高斯相近。")

# ══════════════════════════════════════════════════════════
# 頁籤二：相變點比較
# ══════════════════════════════════════════════════════════
with tab_phase:
    st.header("實驗 A：三種分布的相變點比較")
    st.markdown("固定 **n=8, q=101, m=16, t=1**，以 ρ_eff 為橫軸，三種分布並排比較。")

    col_c, col_r2 = st.columns([1, 2])
    with col_c:
        phase_trials = st.slider("每點測試次數", 200, 1000, 500, step=100, key="ph_tr")
        run_phase = st.button("▶️ 執行實驗 A", type="primary", use_container_width=True)
        st.caption("掃描 ρ_eff 從 0.05 到 1.20，步長 0.05\n共 24 個數據點 × 3 種分布")

    with col_r2:
        # 嘗試載入已存結果
        result_file = "results3/expA_phase_transition.json"
        if os.path.exists(result_file) and not run_phase:
            with open(result_file, encoding="utf-8") as f:
                phaseA = json.load(f)
            st.info("✅ 已載入本地結果（results3/expA_phase_transition.json）")
        elif run_phase:
            phaseA = []
            prog = st.progress(0)
            rho_targets = [round(x, 2) for x in np.arange(0.05, 1.25, 0.05)]
            for i, rho_t in enumerate(rho_targets):
                k   = max(1, round(rho_t * (101/4) / np.sqrt(16/6)))
                sg  = max(0.5, rho_t * (101/4) / np.sqrt(16/2))
                eta = max(1, round(4 * (rho_t * 101/4)**2 / 16))
                row = {"rho_eff_target": rho_t}
                for d, p, pfx in [("uniform",k,"uniform"), ("gaussian",sg,"gauss"), ("cbd",eta,"cbd")]:
                    r = measure_error_rate(8, 101, 16, d, p, trials=phase_trials, seed=42+i*10)
                    row[f"{pfx}_error"] = r["error_rate"]
                    row[f"{pfx}_rho_eff"] = r["rho_eff"]
                phaseA.append(row)
                prog.progress((i+1)/len(rho_targets))
            st.success("✅ 實驗 A 完成！")
        else:
            st.info("按「▶️ 執行實驗 A」，或將 results3/ 資料夾放在同目錄下自動載入。")
            phaseA = None

        if phaseA and PLOTLY:
            import pandas as pd
            df = pd.DataFrame(phaseA)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["rho_eff_target"],
                y=df["uniform_error"]*100, name="均勻 U(-k,k)",
                line=dict(color="#EF4444", width=2)))
            fig.add_trace(go.Scatter(x=df["rho_eff_target"],
                y=df["gauss_error"]*100, name="高斯 DG(σ)",
                line=dict(color="#2E75B6", width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=df["rho_eff_target"],
                y=df["cbd_error"]*100, name="CBD(η)",
                line=dict(color="#16A34A", width=2, dash="dot")))
            fig.add_hline(y=0.5, line_dash="longdash",
                          annotation_text="0.5% 安全門檻", line_color="orange")
            fig.update_layout(
                xaxis_title="有效雜訊比 ρ_eff",
                yaxis_title="解密錯誤率 (%)",
                height=400, legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("安全邊界排序：CBD ≈ 高斯 > 均勻。均勻分布因厚尾效應在相變區錯誤率系統性偏高。")

# ══════════════════════════════════════════════════════════
# 頁籤三：安全性地圖
# ══════════════════════════════════════════════════════════
with tab_map:
    st.header("實驗 B：三張安全性地圖")
    st.markdown("固定 **n=8, m=16, t=1**，對 q ∈ 質數[17,127] 建立三張二維熱圖。")

    dist_map = st.radio("選擇分布", ["uniform","gaussian","cbd"],
        format_func=lambda x: {"uniform":"均勻","gaussian":"高斯","cbd":"CBD"}[x],
        horizontal=True, key="map_dist")

    map_file = f"results3/expB_map_{dist_map}.json"
    if os.path.exists(map_file):
        with open(map_file, encoding="utf-8") as f:
            map_data = json.load(f)
        st.success(f"✅ 已載入 {map_file}（{len(map_data)} 筆數據）")

        if PLOTLY:
            import pandas as pd
            df_map = pd.DataFrame(map_data)
            fig_map = go.Figure(data=go.Heatmap(
                x=df_map["q"],
                y=df_map["rho_eff"].round(2),
                z=df_map["error_rate"] * 100,
                colorscale="RdYlGn_r",
                colorbar=dict(title="錯誤率(%)"),
                zmin=0, zmax=50,
            ))
            param_name = {"uniform":"k","gaussian":"σ","cbd":"η"}[dist_map]
            fig_map.update_layout(
                title=f"LWE 安全性地圖（{dist_map} 分布）",
                xaxis_title="模數 q",
                yaxis_title=f"有效雜訊比 ρ_eff",
                height=480,
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("綠色 = 安全（低錯誤率），紅色 = 危險（高錯誤率）。安全邊界為顏色轉變的分界線。")

            # 各分布的安全邊界摘要
            st.subheader("安全邊界摘要")
            boundary = {
                "uniform": {"安全區 ρ_eff ≤": "0.207", "均值": "0.263", "標準差": "0.056"},
                "gaussian": {"安全區 ρ_eff ≤": "0.256", "均值": "0.303", "標準差": "0.047"},
                "cbd": {"安全區 ρ_eff ≤": "0.320", "均值": "0.343", "標準差": "0.023 ★最佳"},
            }
            st.table(boundary)
    else:
        st.info("請先執行 `python lwe_three_dist.py --exp B` 產生數據，再放到 results3/ 資料夾。")

# ══════════════════════════════════════════════════════════
# 頁籤四：縮小版 Kyber
# ══════════════════════════════════════════════════════════
with tab_kyber:
    st.header("實驗 C：縮小版 Kyber 參數搜尋")
    st.markdown("固定 **n=16, q=257（費馬質數 2⁸+1）, m=32, t=1**，三種分布各自搜尋最大安全參數。")

    kyber_file = "results3/expC_kyber.json"
    if os.path.exists(kyber_file):
        with open(kyber_file, encoding="utf-8") as f:
            kyber_data = json.load(f)
        st.success(f"✅ 已載入 {kyber_file}（{len(kyber_data)} 筆數據）")

        if PLOTLY:
            import pandas as pd
            df_k = pd.DataFrame(kyber_data)
            colors = {"uniform": "#EF4444", "gaussian": "#2E75B6", "cbd": "#16A34A"}
            fig_k = go.Figure()
            for d in ["uniform", "gaussian", "cbd"]:
                dd = df_k[df_k["dist"] == d]
                name = {"uniform":"均勻","gaussian":"高斯","cbd":"CBD"}[d]
                fig_k.add_trace(go.Scatter(
                    x=dd["rho_eff"], y=dd["error_rate"]*100,
                    name=name, mode="lines+markers",
                    line=dict(color=colors[d], width=2)))
            fig_k.add_hline(y=0.1, line_dash="dash",
                            annotation_text="0.1% 目標門檻", line_color="orange")
            fig_k.update_layout(
                xaxis_title="有效雜訊比 ρ_eff",
                yaxis_title="解密錯誤率 (%)",
                height=380)
            st.plotly_chart(fig_k, use_container_width=True)

        # 摘要表
        st.subheader("最大安全參數對比")
        summary = [
            {"分布": "均勻 U(-k,k)", "最大安全參數": "k=8", "ρ_eff": "0.2876",
             "錯誤率": "0.050%", "說明": "和均勻版實驗五一致"},
            {"分布": "高斯 DG(σ)", "最大安全參數": "σ=5.0", "ρ_eff": "0.3113",
             "錯誤率": "0.050%", "說明": "安全邊界比均勻寬 8%"},
            {"分布": "CBD(η)", "最大安全參數": "η=19", "ρ_eff": "0.1919",
             "錯誤率": "0.000%", "說明": "η 整數限制，未達最優"},
            {"分布": "Kyber-512（參考）", "最大安全參數": "η=3", "ρ_eff": "0.0000236",
             "錯誤率": "≈0%", "說明": "n=256 提供主要安全性"},
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
        st.caption("Kyber-512 的 ρ_eff 比縮小版小 1000 倍以上——說明 n=256 才是安全性的主要來源。")
    else:
        st.info("請先執行 `python lwe_three_dist.py --exp C` 產生數據。")

# ══════════════════════════════════════════════════════════
# 頁籤五：UTF-8 效率
# ══════════════════════════════════════════════════════════
with tab_utf8:
    st.header("三種分布的 UTF-8 中文加密效率")

    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        user_text = st.text_input("輸入任意中文訊息", value="量子電腦威脅密碼學安全")
    with col_u2:
        enc_choice = st.selectbox("編碼方式", ["utf-8", "big5"], key="utf8_enc")

    if st.button("🌏 三種分布同時加密比較", type="primary", use_container_width=True):
        params_utf8 = [
            ("uniform", 2, "均勻 k=2"),
            ("gaussian", 1.5, "高斯 σ=1.5"),
            ("cbd", 3, "CBD η=3（Kyber 設定）"),
        ]
        rows = []
        for d, p, lbl in params_utf8:
            try:
                r = encrypt_text(n_val, q_val, m_val, d, p, user_text,
                                 encoding=enc_choice, seed=42)
                rows.append({
                    "分布": lbl,
                    "ρ_eff": f"{r['rho_eff']:.4f}",
                    "加密次數": r["encrypt_count"],
                    "加密時間": f"{r['encrypt_ms']:.2f}ms",
                    "解密時間": f"{r['decrypt_ms']:.2f}ms",
                    "還原正確": "✓" if r["correct"] else "✗",
                })
            except Exception as e:
                rows.append({"分布": lbl, "錯誤": str(e)})

        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"文字：「{user_text}」（{len(user_text.encode(enc_choice,'ignore'))} bytes，"
                   f"{len(user_text.encode(enc_choice,'ignore'))*8} bits）")

# ══════════════════════════════════════════════════════════
# 頁籤六：載入結果
# ══════════════════════════════════════════════════════════
with tab_load:
    st.header("載入已存實驗結果")
    st.markdown("將 `results3/` 資料夾中的 JSON 檔上傳查看圖表，或上傳自訂 JSON 檔。")

    uploaded = st.file_uploader("上傳 JSON 結果檔", type="json")
    data = None
    if uploaded:
        data = json.load(uploaded)
    elif os.path.exists("results3"):
        files = [f for f in os.listdir("results3") if f.endswith(".json")]
        if files:
            sel = st.selectbox("或選擇本地結果", files)
            with open(f"results3/{sel}", encoding="utf-8") as f:
                data = json.load(f)

    if data:
        import pandas as pd
        df = pd.DataFrame(data)
        # 大整數轉字串
        for col in ["estimated_total", "attack_attempts"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"{int(x):,}" if x and str(x) not in ("None","nan") else "─"
                )
        st.subheader(f"數據：{len(df)} 筆")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if PLOTLY and "error_rate" in df.columns:
            x_opts = [c for c in df.columns if c not in ("error_rate",)]
            x_col = st.selectbox("橫軸", x_opts,
                index=x_opts.index("rho_eff") if "rho_eff" in x_opts else 0)
            fig_l = go.Figure()
            if "dist" in df.columns:
                for d in df["dist"].unique():
                    dd = df[df["dist"] == d]
                    name = {"uniform":"均勻","gaussian":"高斯","cbd":"CBD"}.get(d, d)
                    fig_l.add_trace(go.Scatter(
                        x=dd[x_col], y=dd["error_rate"]*100,
                        name=name, mode="lines+markers"))
            else:
                fig_l.add_trace(go.Scatter(
                    x=df[x_col], y=df["error_rate"]*100,
                    name="解密錯誤率", mode="lines+markers"))
            if "theory_pfail" in df.columns:
                fig_l.add_trace(go.Scatter(
                    x=df[x_col], y=df["theory_pfail"].fillna(0)*100,
                    name="理論預測", mode="lines",
                    line=dict(dash="dot", color="#888888")))
            fig_l.update_layout(
                xaxis_title=x_col,
                yaxis_title="解密錯誤率 (%)", height=400)
            st.plotly_chart(fig_l, use_container_width=True)
