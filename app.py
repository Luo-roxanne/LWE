"""
app.py  ── Streamlit 互動展示介面（七個實驗完整版）
─────────────────────────────────────────────────────────────────────────────
旺宏科學獎：格密碼 LWE 安全性參數空間分析

頁籤：
  🔒 加解密示範     即時展示 LWE 加解密（含 t bits 與 UTF-8）
  📐 實驗一         維度基準線
  📈 實驗二         相變點定位
  🗺  實驗三         安全性地圖
  📊 實驗四         公鑰行數影響
  🔬 實驗五         縮小版 Kyber
  ⚡ 實驗六         t bits 影響
  🌏 實驗七         UTF-8 中文加密
  📂 載入結果       視覺化已存數據

執行：streamlit run app.py
"""

import json
import os
import time

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="LWE 安全性分析｜旺宏科學獎",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from lwe_core import (
        LWEParams, keygen, encrypt, decrypt,
        measure_error_rate, brute_force_attack,
        noise_ratio, theoretical_fail_rate,
        validate_params, encode_symbol, decode_symbol,
        encrypt_text_timed, encrypt_text, decrypt_text,
    )
except ImportError as e:
    st.error(f"無法匯入 lwe_core.py：{e}")
    st.stop()

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

MAX_BRUTE_FORCE = 500_000_000
ATTACK_TIMEOUT  = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# 側邊欄
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔐 LWE 參數設定")
    st.markdown("---")

    n_val = st.slider("維度 n", 2, 20, 4,
        help="秘密向量維度，安全性主要來源。n=1 無密碼學意義（毫秒可破）")
    q_val = st.selectbox("模數 q（質數）",
        [17, 23, 31, 41, 53, 71, 97, 101, 127, 257], index=7,
        help="輔助模數（角色不同於 ECC 的 p，安全性主要來自 n 而非 q）")
    k_val = st.slider("雜訊幅度 k", 1, max(1, q_val // 4 + 3),
        min(2, q_val // 8),
        help="雜訊從 U[-k, k] 均勻採樣")
    t_val = st.slider("每次加密 bit 數 t", 1, 5, 1,
        help="t=1：基本版。t>1：每次加密更多 bit，但安全邊界縮窄")
    m_val = 2 * n_val

    st.markdown("---")
    st.subheader("📊 安全性指標")

    rho   = noise_ratio(k_val, q_val, t=1)
    rho_t = noise_ratio(k_val, q_val, t=t_val)
    warn  = validate_params(n_val, q_val, k_val, m_val, t_val)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("雜訊比 ρ", f"{rho:.3f}",
            help="ρ = k/(q/4)，t=1 時的基本安全指標")
    with c2:
        st.metric("廣義 ρ_t", f"{rho_t:.3f}",
            help=f"ρ_t = k/(q/2^(t+1))，納入 t={t_val} 的影響")

    if rho_t < 0.5:
        st.success(f"✅ 安全區（ρ_t={rho_t:.3f} << 1）")
    elif rho_t < 0.9:
        st.warning(f"⚠️ 邊界區（ρ_t={rho_t:.3f} ≈ 0.5-0.9）")
    else:
        st.error(f"❌ 危險區（ρ_t={rho_t:.3f} ≥ 0.9）")

    try:
        theory = theoretical_fail_rate(k_val, q_val, m_val, t_val)
        st.metric("理論失敗率", f"{theory*100:.4f}%")
    except Exception:
        pass

    if warn:
        st.warning(warn)

    st.markdown("---")
    st.caption(
        f"m = 2n = {m_val}（自動設定）\n\n"
        f"q^n = {q_val}^{n_val} = {q_val**n_val:,}\n\n"
        "**公開矩陣 A**：完全隨機產生\n"
        "（安全性假設最少，適合研究）"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 九個頁籤
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🔒 加解密示範",
    "📐 實驗一：維度",
    "📈 實驗二：相變點",
    "🗺 實驗三：地圖",
    "📊 實驗四：m 影響",
    "🔬 實驗五：Kyber",
    "⚡ 實驗六：t bits",
    "🌏 實驗七：UTF-8",
    "📂 載入結果",
])

tab_demo, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab_load = tabs


# ══════════════════════════════════════════════════════════════════════════════
# 頁籤：加解密示範
# ══════════════════════════════════════════════════════════════════════════════

with tab_demo:
    st.header("LWE 加解密完整示範")
    st.caption(f"n={n_val}, q={q_val}, k={k_val}, m={m_val}, t={t_val}")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("① 金鑰生成")
        seed = st.number_input("隨機種子", value=42, step=1)
        if st.button("🔑 生成金鑰對", type="primary", use_container_width=True):
            try:
                params = LWEParams(n=n_val, q=q_val, k=k_val, m=m_val, t=t_val)
                rng = np.random.default_rng(int(seed))
                pub, priv = keygen(params, rng=rng)
                st.session_state.update({"pub": pub, "priv": priv,
                                         "seed": int(seed)})
                st.success("金鑰生成成功！")
            except ValueError as e:
                st.error(str(e))

        pub  = st.session_state.get("pub")
        priv = st.session_state.get("priv")

        if pub and priv:
            with st.expander("🔍 私鑰 s（實際中絕不公開）"):
                st.code(str(priv.s.tolist()))
            with st.expander("🔍 公開矩陣 A（前 3 行）"):
                st.code(str(pub.A[:3].tolist()))
                st.caption("A 完全公開，每個元素從 0 到 q-1 均勻隨機選取")
            with st.expander("🔍 公鑰向量 b = As + e (mod q)"):
                st.code(str(pub.b.tolist()))

    with col_r:
        st.subheader(f"② 加密（t={t_val} bits）")
        symbols = 2 ** t_val
        mu_input = st.selectbox(
            f"選擇符號 μ（0 到 {symbols-1}，共 {symbols} 種）",
            list(range(symbols)),
            help=f"t={t_val} 時每次加密 {t_val} bits，可表示 {symbols} 個符號"
        )
        if st.button("🔒 加密", type="primary", use_container_width=True):
            if pub:
                try:
                    rng2 = np.random.default_rng(
                        st.session_state.get("seed", 42) + 1)
                    ct = encrypt(pub, mu_input, rng=rng2)
                    st.session_state["ct"] = ct
                    st.session_state["mu_sent"] = mu_input
                    st.success(f"加密完成！μ = {mu_input} = "
                               f"{format(mu_input, f'0{t_val}b')}₂")
                except Exception as e:
                    st.error(str(e))
            else:
                st.error("請先生成金鑰")

        ct = st.session_state.get("ct")
        if ct:
            with st.expander("🔍 密文 (u, v)"):
                st.code(f"u = {ct.u.tolist()}\nv = {ct.v}")
                qn = q_val ** n_val
                st.caption(f"Eve 看到 (u, v) 但需嘗試 q^n = {qn:,} 種可能才能破解")

    st.divider()
    st.subheader("③ 解密")
    if ct and priv:
        if st.button("🔓 解密", use_container_width=False):
            mu_recv = decrypt(priv, ct)
            st.session_state["mu_recv"] = mu_recv
        if "mu_recv" in st.session_state:
            mu_sent = st.session_state.get("mu_sent")
            mu_recv = st.session_state["mu_recv"]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("原始符號 μ", f"{mu_sent} = {format(mu_sent, f'0{t_val}b')}₂")
            with c2:
                st.metric("解密結果 μ̂", f"{mu_recv} = {format(mu_recv, f'0{t_val}b')}₂")
            with c3:
                st.metric("結果", "✅ 正確" if mu_recv == mu_sent else "❌ 錯誤")

            with st.expander("📐 數學步驟"):
                q = priv.params.q
                t = ct.t
                d_raw = ct.v - int(ct.u @ priv.s)
                d = d_raw % q
                expected_center = encode_symbol(mu_sent, q, t)
                st.code(
                    f"d = v - u·s (mod q)\n"
                    f"  = {ct.v} - {int(ct.u @ priv.s) % q} (mod {q})\n"
                    f"  = {d}\n\n"
                    f"μ={mu_sent} 的區間中心 = {expected_center}\n"
                    f"解碼結果：{decode_symbol(d, q, t)}\n"
                    f"正確性條件：|rᵀe| < q/2^(t+1) = {q}/{2**(t+1)} = {q/2**(t+1):.1f}"
                )
    else:
        st.info("請先完成金鑰生成與加密")

    st.divider()
    st.subheader("④ UTF-8 中文訊息加密示範")
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        user_text = st.text_input("輸入任意文字（支援中文）", value="你好世界")
        enc_choice = st.selectbox("編碼方式", ["utf-8", "ascii", "big5"])
    with col_u2:
        st.metric("Bytes", f"{len(user_text.encode(enc_choice, errors='ignore'))}")
        st.metric("Bits", f"{len(user_text.encode(enc_choice, errors='ignore')) * 8}")
        st.metric("加密次數（t=t_val）",
                  f"{-(-len(user_text.encode(enc_choice, errors='ignore')) * 8 // t_val)}")

    if st.button("🌏 加密並解密", use_container_width=False):
        try:
            params = LWEParams(n=n_val, q=q_val, k=k_val, m=m_val, t=t_val)
            with st.spinner("加密中..."):
                stats = encrypt_text_timed(params, user_text,
                                           encoding=enc_choice, seed=42)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("加密次數", stats.encrypt_count)
            with c2:
                st.metric("加密時間", f"{stats.encrypt_time*1000:.1f}ms")
            with c3:
                st.metric("解密時間", f"{stats.decrypt_time*1000:.1f}ms")
            with c4:
                correct = stats.recovered_text == user_text
                st.metric("還原正確", "✅" if correct else "❌")
            if not correct:
                st.error(f"還原結果：「{stats.recovered_text}」")
                st.warning(f"提示：ρ_t={rho_t:.3f}，接近相變點可能導致解密失敗。"
                           f"請降低 k 或減小 t。")
        except ValueError as e:
            st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 實驗一
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("實驗一：維度基準線")
    st.markdown(
        "固定 **q=101, k=2, t=1, m=2n**，掃描維度 n，"
        "觀察破解時間的指數增長與解密錯誤率的穩定。"
    )
    col_cfg, col_res = st.columns([1, 2])
    with col_cfg:
        e1_nmax    = st.slider("最大維度 n_max", 4, 15, 8, key="e1_nmax")
        e1_trials  = st.slider("每 n 測試次數", 100, 1000, 300, step=100, key="e1_tr")
        e1_timeout = st.slider("攻擊超時（秒）", 5, 60, 30, key="e1_to")
        st.caption(
            f"q=101, k=2, t=1  |  ρ={noise_ratio(2,101):.3f}\n\n"
            f"跳過條件：q^n > {MAX_BRUTE_FORCE:,}"
        )
        run1 = st.button("▶️ 開始實驗一", type="primary", use_container_width=True)

    with col_res:
        if run1:
            import pandas as pd
            results1 = []
            prog = st.progress(0)
            status = st.empty()
            chart_ph = st.empty()
            table_ph = st.empty()
            ns = list(range(2, e1_nmax + 1))

            for i, n in enumerate(ns):
                m = 2 * n
                params = LWEParams(n=n, q=101, k=2, m=m, t=1)
                status.text(f"n={n}：解密錯誤率測試（{e1_trials} 次）...")
                err = measure_error_rate(params, trials=e1_trials, seed=42+i)

                est = 101 ** n
                atk_elapsed = atk_success = None
                if est <= MAX_BRUTE_FORCE:
                    status.text(f"n={n}：暴力攻擊（q^n={est:,}）...")
                    rng = np.random.default_rng(42 + i)
                    pub_a, _ = keygen(params, rng=rng)
                    from lwe_core import brute_force_attack as bfa
                    atk = bfa(pub_a, timeout=e1_timeout)
                    atk_elapsed, atk_success = atk.elapsed, atk.success

                results1.append({
                    "n": n, "m": m, "estimated": est,
                    "error_rate": err["error_rate"],
                    "atk_elapsed": atk_elapsed,
                    "atk_success": atk_success,
                })
                prog.progress((i+1)/len(ns))

                if PLOTLY and len(results1) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[r["n"] for r in results1],
                        y=[r["error_rate"]*100 for r in results1],
                        name="解密錯誤率(%)", line=dict(color="#2E75B6")))
                    atk_data = [(r["n"], r["atk_elapsed"])
                                for r in results1 if r["atk_elapsed"]]
                    if atk_data:
                        fig.add_trace(go.Scatter(
                            x=[d[0] for d in atk_data],
                            y=[d[1] for d in atk_data],
                            name="攻擊時間(s)", yaxis="y2",
                            line=dict(color="#E05252", dash="dash")))
                    fig.update_layout(
                        xaxis_title="維度 n",
                        yaxis=dict(title="解密錯誤率(%)", range=[-5, 105]),
                        yaxis2=dict(title="攻擊時間(s)", overlaying="y",
                                    side="right", type="log"),
                        height=350, legend=dict(x=0.01, y=0.99))
                    chart_ph.plotly_chart(fig, use_container_width=True)

            status.success("✅ 實驗一完成！")
            st.session_state["exp1"] = results1

            df = pd.DataFrame(results1)
            df["q^n"] = df["estimated"].map(lambda x: f"{x:,}")
            df["錯誤率"] = df["error_rate"].map(lambda x: f"{x*100:.2f}%")
            df["攻擊"] = df.apply(lambda r:
                "✓ 成功" if r["atk_success"] is True
                else ("✗ 超時" if r["atk_success"] is False else "跳過"), axis=1)
            df["時間"] = df["atk_elapsed"].map(
                lambda x: f"{x:.3f}s" if x else "─")
            table_ph.dataframe(
                df[["n","m","q^n","錯誤率","攻擊","時間"]],
                hide_index=True, use_container_width=True)
        else:
            st.info("調整左側設定後按「▶️ 開始實驗一」")


# ══════════════════════════════════════════════════════════════════════════════
# 實驗二
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("實驗二：相變點精確定位")
    st.markdown(
        "固定 **n=8, q=101, t=1, m=16**，掃描雜訊幅度 k，"
        "觀察解密錯誤率在 ρ≈1（k≈25）時的急劇相變，"
        "並對比理論預測曲線（常態近似）。"
    )

    col_c, col_r2 = st.columns([1, 2])
    with col_c:
        e2_trials = st.slider("每 k 測試次數", 200, 2000, 500,
                              step=100, key="e2_tr")
        st.caption("固定：n=8, q=101, t=1, m=16\n理論相變點：k ≈ q/4 ≈ 25")
        run2 = st.button("▶️ 開始實驗二", type="primary", use_container_width=True)

    with col_r2:
        if run2:
            import pandas as pd
            results2 = []
            prog2 = st.progress(0)
            chart2 = st.empty()
            k_range = list(range(1, 29))

            for i, k in enumerate(k_range):
                params = LWEParams(n=8, q=101, k=k, m=16, t=1)
                err = measure_error_rate(params, trials=e2_trials, seed=42+k)
                results2.append(err)
                prog2.progress((i+1)/len(k_range))

                if PLOTLY and len(results2) > 1:
                    ks   = [r["k"]          for r in results2]
                    errs = [r["error_rate"]*100 for r in results2]
                    th   = [(r["theory_fail"] or 0)*100 for r in results2]
                    rhos = [r["rho"]         for r in results2]
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=ks, y=errs, name="實驗錯誤率",
                        line=dict(color="#2E75B6", width=2)))
                    fig2.add_trace(go.Scatter(
                        x=ks, y=th, name="理論預測",
                        line=dict(color="#E05252", dash="dot")))
                    fig2.add_vline(x=25, line_dash="dash",
                                   annotation_text="理論相變點 k≈25")
                    fig2.update_layout(
                        xaxis_title="雜訊幅度 k",
                        yaxis_title="解密錯誤率(%)",
                        height=380, legend=dict(x=0.01, y=0.99))
                    chart2.plotly_chart(fig2, use_container_width=True)

            st.session_state["exp2"] = results2
            st.success("✅ 實驗二完成！")

            df2 = pd.DataFrame(results2)
            df2["錯誤率"] = df2["error_rate"].map(lambda x: f"{x*100:.2f}%")
            df2["理論"] = df2["theory_fail"].map(
                lambda x: f"{(x or 0)*100:.3f}%")
            df2["ρ"] = df2["rho"].map(lambda x: f"{x:.3f}")
            st.dataframe(df2[["k","ρ","錯誤率","理論"]],
                         hide_index=True, use_container_width=True)
        else:
            st.info("按「▶️ 開始實驗二」")


# ══════════════════════════════════════════════════════════════════════════════
# 實驗三
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("實驗三：安全性地圖（二維熱圖）")
    st.markdown(
        "固定 **n=8, t=1, m=16**，對所有 (q, k) 組合測量解密錯誤率，"
        "繪製二維熱圖。預期安全邊界為 **ρ = k/(q/4) ≈ 1** 的等值線。"
    )

    col_c3, col_r3 = st.columns([1, 2])
    with col_c3:
        e3_trials = st.slider("每組測試次數", 100, 500, 200,
                              step=50, key="e3_tr")
        e3_qmax   = st.slider("最大 q", 53, 127, 71, key="e3_qm")
        st.caption("⚠ 完整掃描需較長時間\n建議先用較小的 q 範圍測試")
        run3 = st.button("▶️ 開始實驗三", type="primary", use_container_width=True)

    with col_r3:
        if run3:
            def sieve(limit):
                is_p = [True]*(limit+1)
                is_p[0] = is_p[1] = False
                for i in range(2, int(limit**0.5)+1):
                    if is_p[i]:
                        for j in range(i*i, limit+1, i): is_p[j] = False
                return [i for i in range(2, limit+1) if is_p[i] and i >= 17]

            q_list = sieve(e3_qmax)
            prog3 = st.progress(0)
            status3 = st.empty()
            results3 = []
            total3 = sum(q//4+5 for q in q_list)
            done3 = 0

            for q in q_list:
                for k in range(1, q//4+6):
                    params = LWEParams(n=8, q=q, k=k, m=16, t=1)
                    err = measure_error_rate(params, trials=e3_trials,
                                            seed=42+q*100+k)
                    results3.append(err)
                    done3 += 1
                    prog3.progress(min(done3/total3, 1.0))
                    status3.text(f"進度：{done3}/{total3}  q={q}, k={k}")

            st.session_state["exp3"] = results3
            status3.success(f"✅ 完成！共 {len(results3)} 組數據")

            if PLOTLY and results3:
                import pandas as pd
                df3 = pd.DataFrame(results3)
                fig3 = go.Figure(data=go.Heatmap(
                    x=df3["q"], y=df3["k"],
                    z=df3["error_rate"]*100,
                    colorscale="RdYlGn_r",
                    colorbar=dict(title="解密錯誤率(%)"),
                ))
                fig3.update_layout(
                    title="LWE 安全性地圖：解密錯誤率 (n=8, t=1)",
                    xaxis_title="模數 q",
                    yaxis_title="雜訊幅度 k",
                    height=450,
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("綠色 = 安全（低錯誤率），紅色 = 危險（高錯誤率）。"
                           "安全邊界應接近 ρ = k/(q/4) = 1 的曲線。")
        else:
            st.info("按「▶️ 開始實驗三」（注意：掃描範圍越大耗時越長）")


# ══════════════════════════════════════════════════════════════════════════════
# 實驗四
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("實驗四：公鑰行數 m 的影響")
    st.markdown(
        "固定 **n=8, q=101, k=2, t=1**，掃描 m 從 n 到 5n，"
        "驗證解密錯誤率與 m 無關（廣義雜訊比 ρ_t 與 m 無關）。"
    )

    col_c4, col_r4 = st.columns([1, 2])
    with col_c4:
        e4_trials = st.slider("每 m 測試次數", 200, 1000, 500, key="e4_tr")
        run4 = st.button("▶️ 開始實驗四", type="primary", use_container_width=True)
        st.caption("固定：n=8, q=101, k=2, t=1\n理論預測：錯誤率不隨 m 改變")

    with col_r4:
        if run4:
            import pandas as pd
            results4 = []
            prog4 = st.progress(0)
            m_range = list(range(8, 41))

            for i, m in enumerate(m_range):
                params = LWEParams(n=8, q=101, k=2, m=m, t=1)
                err = measure_error_rate(params, trials=e4_trials, seed=42+m)
                results4.append(err)
                prog4.progress((i+1)/len(m_range))

            st.session_state["exp4"] = results4
            st.success("✅ 實驗四完成！")

            if PLOTLY:
                df4 = pd.DataFrame(results4)
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=df4["m"], y=df4["error_rate"]*100,
                    mode="lines+markers", name="解密錯誤率",
                    line=dict(color="#2E75B6")))
                fig4.add_vline(x=16, line_dash="dash",
                               annotation_text="m=2n=16（建議值）")
                fig4.update_layout(
                    xaxis_title="公鑰行數 m",
                    yaxis_title="解密錯誤率(%)",
                    height=350)
                st.plotly_chart(fig4, use_container_width=True)
                st.caption("若錯誤率幾乎不隨 m 改變，驗證理論預測正確。")
        else:
            st.info("按「▶️ 開始實驗四」")


# ══════════════════════════════════════════════════════════════════════════════
# 實驗五
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("實驗五：縮小版 Kyber 參數搜尋")
    st.markdown(
        "固定 **n=16, q=257（費馬質數 2⁸+1）, t=1**，搜尋最大的安全 k。"
        "對比 Kyber-512（n=256, q=3329）的參數設計邏輯。"
    )

    col_c5, col_r5 = st.columns([1, 2])
    with col_c5:
        e5_trials = st.slider("每 k 測試次數", 500, 3000, 1000, key="e5_tr")
        run5 = st.button("▶️ 開始實驗五", type="primary", use_container_width=True)
        st.caption(
            "固定：n=16, q=257, m=32, t=1\n"
            "目標：錯誤率 < 0.1%\n"
            "費馬質數 257 = 2⁸+1（計算效率優良）"
        )

    with col_r5:
        if run5:
            import pandas as pd
            results5 = []
            prog5 = st.progress(0)
            k_range5 = list(range(1, 257//4+1))

            for i, k in enumerate(k_range5):
                params = LWEParams(n=16, q=257, k=k, m=32, t=1)
                err = measure_error_rate(params, trials=e5_trials, seed=42+k)
                err["target_met"] = err["error_rate"] < 0.001
                results5.append(err)
                prog5.progress((i+1)/len(k_range5))
                if err["error_rate"] >= 0.05:
                    break

            st.session_state["exp5"] = results5
            safe_ks = [r for r in results5 if r["target_met"]]
            best_k5 = safe_ks[-1]["k"] if safe_ks else None

            st.success(f"✅ 完成！最佳 k = {best_k5}"
                       f"（ρ = {noise_ratio(best_k5, 257):.4f}）" if best_k5
                       else "✅ 完成！")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("最大安全 k", best_k5 or "─")
                if best_k5:
                    st.metric("對應 ρ", f"{noise_ratio(best_k5, 257):.4f}")
            with c2:
                st.metric("Kyber-512 ρ（估算）", "~0.004")
                st.caption("Kyber 用 n=256 而非大 k 提供安全性")

            if PLOTLY:
                df5 = pd.DataFrame(results5)
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(
                    x=df5["k"], y=df5["error_rate"]*100,
                    name="錯誤率", line=dict(color="#2E75B6")))
                fig5.add_hline(y=0.1, line_dash="dash",
                               annotation_text="0.1% 目標線")
                fig5.update_layout(xaxis_title="k",
                                   yaxis_title="解密錯誤率(%)", height=350)
                st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("按「▶️ 開始實驗五」")


# ══════════════════════════════════════════════════════════════════════════════
# 實驗六
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header("實驗六：每次加密 bit 數 t 的影響")
    st.markdown(
        "固定 **n=8, q=101, k=2, m=16**，掃描 t 從 1 到 5。\n\n"
        "核心結論：**增大 t = 零和遊戲**（效率↑，安全性↓）；"
        "**增大 n = 兩全其美**（效率↑，安全性↑）。"
    )

    col_c6, col_r6 = st.columns([1, 2])
    with col_c6:
        e6_trials = st.slider("每 t 測試次數", 200, 1000, 500, key="e6_tr")
        run6 = st.button("▶️ 開始實驗六", type="primary", use_container_width=True)
        st.caption(
            "固定：n=8, q=101, k=2, m=16\n"
            "掃描：t = 1 到 5\n"
            "一個中文字 = 24 bits = 3 bytes"
        )

    with col_r6:
        if run6:
            import pandas as pd
            results6 = []
            for t in range(1, 6):
                if 2**t > 101:
                    break
                params = LWEParams(n=8, q=101, k=2, m=16, t=t)
                err = measure_error_rate(params, trials=e6_trials, seed=42+t)
                enc_count = -(-24 // t)
                results6.append({
                    **err,
                    "symbols": 2**t,
                    "interval_w": 101 / 2**t,
                    "enc_count_24bits": enc_count,
                    "safe": err["error_rate"] < 0.001,
                })

            st.session_state["exp6"] = results6
            st.success("✅ 實驗六完成！")

            df6 = pd.DataFrame(results6)
            safe_t = max([r["t"] for r in results6 if r["safe"]], default=1)
            st.info(f"在 n=8, q=101, k=2 下，t 的最大安全值 = **{safe_t}**")

            if PLOTLY:
                fig6 = go.Figure()
                fig6.add_trace(go.Bar(
                    x=df6["t"], y=df6["error_rate"]*100,
                    name="解密錯誤率(%)",
                    marker_color=["#2E75B6" if r["safe"] else "#E05252"
                                  for r in results6]))
                fig6.add_hline(y=0.1, line_dash="dash",
                               annotation_text="0.1% 目標線")
                fig6.update_layout(
                    xaxis_title="每次加密 bit 數 t",
                    yaxis_title="解密錯誤率(%)",
                    height=350)
                st.plotly_chart(fig6, use_container_width=True)

            df6["錯誤率"] = df6["error_rate"].map(lambda x: f"{x*100:.2f}%")
            df6["ρ_t"] = df6["rho_t"].map(lambda x: f"{x:.3f}")
            df6["安全"] = df6["safe"].map(lambda x: "✓" if x else "✗")
            df6["加密中文字次數"] = df6["enc_count_24bits"]
            st.dataframe(
                df6[["t","symbols","ρ_t","錯誤率","安全","加密中文字次數"]],
                hide_index=True, use_container_width=True)
            st.caption("藍色 = 安全（錯誤率<0.1%），紅色 = 超標。"
                       "加密次數減少代表效率提升，但安全邊界縮窄。")
        else:
            st.info("按「▶️ 開始實驗六」")


# ══════════════════════════════════════════════════════════════════════════════
# 實驗七
# ══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.header("實驗七：UTF-8 中文訊息端對端加密效率")
    st.markdown(
        "測試不同參數組合對中文訊息加密效率的影響。"
        "「加密一個中文字需要多久？對應的安全性是多少？」"
    )

    col_c7, col_r7 = st.columns([1, 2])
    with col_c7:
        e7_text = st.text_area("測試文字", value="量子電腦威脅密碼學安全\n你好世界\nHello LWE")
        e7_enc  = st.selectbox("編碼方式", ["utf-8", "big5", "ascii"], key="e7_enc")
        run7 = st.button("▶️ 開始實驗七", type="primary", use_container_width=True)

    with col_r7:
        if run7:
            import pandas as pd
            configs = [
                (4, 101, 2,  8, 1, "低維 t=1"),
                (4, 101, 2,  8, 2, "低維 t=2"),
                (8, 101, 2, 16, 1, "中維 t=1（基準）"),
                (8, 101, 2, 16, 2, "中維 t=2"),
                (8, 257, 3, 16, 1, "中維 q=257"),
                (16, 257, 3, 32, 1, "高維（迷你Kyber）"),
            ]
            texts7 = [t.strip() for t in e7_text.strip().split("\n") if t.strip()]
            results7 = []

            for text in texts7:
                st.markdown(f"**「{text}」**")
                rows = []
                for n, q, k, m, t, label in configs:
                    try:
                        params = LWEParams(n=n, q=q, k=k, m=m, t=t)
                        stats = encrypt_text_timed(params, text,
                                                   encoding=e7_enc, seed=42)
                        correct = stats.recovered_text == text
                        rows.append({
                            "參數": label,
                            "加密次數": stats.encrypt_count,
                            "加密(ms)": f"{stats.encrypt_time*1000:.1f}",
                            "解密(ms)": f"{stats.decrypt_time*1000:.1f}",
                            "ρ": f"{params.rho:.3f}",
                            "ρ_t": f"{params.rho_t:.3f}",
                            "正確": "✓" if correct else "✗",
                        })
                        results7.append({
                            "text": text, "label": label,
                            "n":n, "q":q, "k":k, "m":m, "t":t,
                            "rho": params.rho, "rho_t": params.rho_t,
                            "encrypt_count": stats.encrypt_count,
                            "encrypt_ms": stats.encrypt_time*1000,
                            "decrypt_ms": stats.decrypt_time*1000,
                            "correct": correct,
                        })
                    except (ValueError, UnicodeEncodeError):
                        pass
                if rows:
                    st.dataframe(pd.DataFrame(rows),
                                 hide_index=True, use_container_width=True)

            st.session_state["exp7"] = results7
            st.success("✅ 實驗七完成！")

            # 編碼比較
            st.markdown("**三種編碼的 byte 數比較：**")
            sample = "你好"
            for enc in ["utf-8", "big5", "ascii"]:
                try:
                    b = sample.encode(enc)
                    st.code(
                        f"{enc:>8}：{sample} → {len(b)} bytes = {len(b)*8} bits"
                        f"（t=1 需 {len(b)*8} 次 LWE 加密）"
                    )
                except (UnicodeEncodeError, LookupError):
                    st.code(f"{enc:>8}：無法編碼「{sample}」")
        else:
            st.info("輸入測試文字後按「▶️ 開始實驗七」")


# ══════════════════════════════════════════════════════════════════════════════
# 載入結果
# ══════════════════════════════════════════════════════════════════════════════

with tab_load:
    st.header("載入已存實驗結果")
    st.markdown(
        "執行 `python experiments.py --exp 1` 後，"
        "將 `results/` 資料夾中的 JSON 上傳到這裡查看圖表。"
    )

    uploaded = st.file_uploader("上傳 JSON 結果檔", type="json")
    data = None

    if uploaded:
        data = json.load(uploaded)
    elif os.path.exists("results"):
        json_files = [f for f in os.listdir("results") if f.endswith(".json")]
        if json_files:
            selected = st.selectbox("或選擇本地結果", json_files)
            with open(f"results/{selected}", encoding="utf-8") as f:
                data = json.load(f)

    if data:
        import pandas as pd
        df = pd.DataFrame(data)
        st.subheader(f"數據：{len(df)} 筆")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if PLOTLY and "error_rate" in df.columns:
            st.subheader("視覺化")
            x_col = st.selectbox("橫軸", df.columns.tolist(),
                                 index=df.columns.tolist().index("n")
                                 if "n" in df.columns else 0)
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(
                x=df[x_col], y=df["error_rate"]*100,
                mode="lines+markers", name="解密錯誤率(%)"))
            if "theory_fail" in df.columns:
                fig_l.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df["theory_fail"].fillna(0)*100,
                    mode="lines", name="理論預測",
                    line=dict(dash="dot", color="#E05252")))
            fig_l.update_layout(
                xaxis_title=x_col,
                yaxis_title="解密錯誤率(%)", height=400)
            st.plotly_chart(fig_l, use_container_width=True)
