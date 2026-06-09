import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

sns.set_theme(style="whitegrid")

from src.store import (
    dataset, col_meta, schema, all_cols, reset,
    target_candidates, group_candidates, datetime_cols,
    id_candidates, binary_cols, num_cols, cat_cols,
)
from src.data_loader import load_data, resolve_column_names, prepare_data, get_group_stats, get_control_value
from src.data_quality import report_data_quality
from src.descriptive_stats import compute_descriptive_stats
from src.hypothesis_testing import run_all_tests
from src.effect_size import compute_all_effect_sizes
from src.logistic_regression import (
    simple_logistic_regression,
    enriched_logistic_regression,
    likelihood_ratio_test,
)
from src.bayesian_analysis import beta_binomial_analysis
from src.segmentation import run_segmentation_analysis
from src.temporal_analysis import run_temporal_analysis
from src.robustness import run_robustness_checks
from statsmodels.stats.power import NormalIndPower

st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton button { border-radius: 8px; font-weight: 600; }
    .stButton button[kind="primary"] { background: linear-gradient(135deg, #6366f1, #4f46e5); border: none; color: white; }
    .stButton button[kind="primary"]:hover { background: linear-gradient(135deg, #4f46e5, #4338ca); }
    .stButton button[kind="secondary"] { border: 1px solid #e2e8f0; }
    div[data-testid="metric-container"] { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    div[data-testid="metric-container"]:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); transition: box-shadow 0.2s; }
    div[data-testid="metric-container"] label { font-size: 0.8rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.03em; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] { font-size: 1.8rem; font-weight: 700; color: #0f172a; }
    .schema-banner { background: linear-gradient(135deg, #f8fafc, #eef2ff); border: 1px solid #e0e7ff; border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 2rem; }
    .schema-banner h3 { margin: 0 0 0.75rem 0; color: #1e293b; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .schema-tag { display: inline-block; background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.4rem 0.8rem; margin: 0.2rem 0.4rem 0.2rem 0; font-size: 0.85rem; }
    .schema-tag strong { color: #4f46e5; }
    .schema-tag.highlight { border-color: #6366f1; background: #eef2ff; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-top: 2rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #e2e8f0; }
    .rec-card { border-radius: 16px; padding: 1.5rem 2rem; }
    .stAlert { border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 0.5rem 1rem; font-weight: 500; }
    p, li { color: #334155; }
    hr { margin: 1.5rem 0; border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
for key in ("analysis_run", "prepared", "results", "data_loaded"):
    if key not in st.session_state:
        st.session_state[key] = False if key in ("analysis_run", "data_loaded") else ({} if key == "results" else None)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>📊 A/B Testing</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 0;'>Professional Analysis Toolkit</p>", unsafe_allow_html=True)
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload dataset", type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Any format with a binary target and a group column",
    )
    use_default = st.checkbox("Use default sample (ab_data.csv)", value=True)

    if st.button("📂 Load Data", type="secondary", use_container_width=True):
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
        elif use_default:
            path = "ab_data.csv"
        else:
            path = None
        if path:
            with st.spinner("Loading data..."):
                df = load_data(Path(path))
                reset(df)
                st.session_state.data_loaded = True
                st.session_state.analysis_run = False
                st.toast(f"Loaded {len(df):,} rows × {len(df.columns)} columns", icon="✅")

    st.divider()

    if not st.session_state.data_loaded:
        st.info("👈 Load data first to configure column mapping")
        st.stop()

    st.markdown("### Column Mapping")

    auto_detect = st.checkbox("Auto-detect columns", value=True)

    default_target = (target_candidates or binary_cols or all_cols or [None])[0]
    default_group = (group_candidates or [None])[0]
    default_id = (id_candidates or [None])[0]
    default_time = (datetime_cols or [None])[0]

    idx_target = all_cols.index(default_target) if default_target and default_target in all_cols else 0
    idx_group = all_cols.index(default_group) if default_group and default_group in all_cols else min(1, len(all_cols)-1)

    if auto_detect:
        selected_target = st.selectbox("🎯 Target (conversion)", all_cols, index=idx_target)
        selected_group = st.selectbox("🔀 Group (treatment)", all_cols, index=idx_group)
    else:
        selected_target = st.selectbox("🎯 Target (conversion)", all_cols)
        selected_group = st.selectbox("🔀 Group (treatment)", all_cols)

    covariate_opts = [c for c in all_cols if c not in (selected_target, selected_group)]
    selected_covariates = st.multiselect("📦 Covariates (optional)", covariate_opts)

    time_opts = ["(None)"] + all_cols
    time_idx = 0
    if default_time and not auto_detect and default_time in all_cols:
        time_idx = all_cols.index(default_time) + 1
    selected_time = st.selectbox("🕐 Time column (optional)", time_opts, index=time_idx)

    id_opts = ["(None)"] + all_cols
    id_idx = 0
    if default_id and not auto_detect and default_id in all_cols:
        id_idx = all_cols.index(default_id) + 1
    selected_id = st.selectbox("🆔 ID column (optional)", id_opts, index=id_idx)

    st.divider()

    alpha = st.slider("Significance level α", 0.01, 0.20, 0.05, 0.01, format="%.2f")

    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ── Pre‑run: Data Preview ─────────────────────────────────────────
if not st.session_state.analysis_run:
    st.markdown("<h1 style='font-size: 2rem; font-weight: 800; color: #0f172a;'>📊 A/B Testing Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.05rem; margin-bottom: 1.5rem;'>Upload any dataset with a binary target and a group column. Auto-detection infers column roles.</p>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2.2, 1])
    with col_left:
        st.markdown("#### Data Preview")
        n_preview = min(10, len(all_cols))
        preview_dict = {col: dataset[col][:10] for col in all_cols[:n_preview]}
        st.dataframe(pd.DataFrame(preview_dict), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("#### Column Types")
        for c in all_cols:
            m = col_meta.get(c, "?")
            icon = "🔢" if m == "numeric" else "🏷️"
            st.markdown(f"{icon} **{c}** → {m}")

    st.divider()

    st.markdown("#### Auto-detected Candidates")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Target", len(target_candidates), ", ".join(target_candidates[:3]) if target_candidates else "—")
    c2.metric("🔀 Group", len(group_candidates), ", ".join(group_candidates[:3]) if group_candidates else "—")
    c3.metric("🕐 Time", len(datetime_cols), ", ".join(datetime_cols[:2]) if datetime_cols else "—")
    c4.metric("🆔 ID", len(id_candidates), ", ".join(id_candidates[:2]) if id_candidates else "—")

    st.info("👈 Configure column mapping in the sidebar and click **▶ Run Analysis**", icon="💡")

    if not run_btn:
        st.stop()

# ── Run Analysis ──────────────────────────────────────────────────
with st.spinner("Running A/B test analysis..."):
    resolve_column_names(
        target_col=selected_target,
        group_col=selected_group,
        covariate_cols=selected_covariates,
        time_col=selected_time if selected_time != "(None)" else "",
        id_col=selected_id if selected_id != "(None)" else "",
    )

    prepared = prepare_data()
    gs = get_group_stats(prepared)
    stats = compute_descriptive_stats(prepared)
    tests = run_all_tests(prepared)
    effect_sizes = compute_all_effect_sizes(prepared)
    bayesian = beta_binomial_analysis(prepared)

    es_val = abs(effect_sizes["cohens_h"])
    if es_val < 1e-9 or gs["n_control"] == 0:
        power_obs, n_needed, mde_h, mde_pp = 0.0, float("inf"), float("nan"), float("nan")
    else:
        try:
            pa = NormalIndPower()
            power_obs = pa.solve_power(effect_size=es_val, nobs1=gs["n_control"], ratio=gs["n_treatment"] / gs["n_control"], alpha=alpha, alternative="two-sided")
            n_needed = pa.solve_power(effect_size=es_val, power=0.80, alpha=alpha, ratio=1.0, alternative="two-sided")
            mde_h = pa.solve_power(nobs1=gs["n_control"], ratio=gs["n_treatment"] / gs["n_control"], power=0.80, alpha=alpha, alternative="two-sided")
            mde_pp = 2 * np.sin(mde_h / 2) * np.sqrt(gs["rate_control"] * (1 - gs["rate_control"]))
        except Exception:
            power_obs, n_needed, mde_h, mde_pp = 0.0, float("inf"), float("nan"), float("nan")

    log_simple = simple_logistic_regression(prepared)
    log_enriched = enriched_logistic_regression(prepared)
    lr_test = likelihood_ratio_test(log_simple["model"], log_enriched["model"])
    segmentation = run_segmentation_analysis(prepared)
    temporal = run_temporal_analysis(prepared)
    robustness = run_robustness_checks(prepared)
    quality = report_data_quality()

    st.session_state.analysis_run = True
    st.session_state.prepared = prepared
    st.session_state.results = {
        "stats": stats, "tests": tests, "effect_sizes": effect_sizes,
        "bayesian": bayesian, "power_obs": power_obs, "n_needed": n_needed,
        "mde_h": mde_h, "mde_pp": mde_pp,
        "log_simple": log_simple, "log_enriched": log_enriched,
        "lr_test": lr_test, "segmentation": segmentation, "temporal": temporal,
        "robustness": robustness, "quality": quality, "gs": gs, "alpha": alpha,
    }

# ── Load Results ──────────────────────────────────────────────────
R = st.session_state.results
stats, tests, es, bayes = R["stats"], R["tests"], R["effect_sizes"], R["bayesian"]
gs, alpha = R["gs"], R["alpha"]
power_obs = R.get("power_obs", 0) or 0

sig_two = tests["z_test"]["p_value_two_sided"] < alpha
sig_one = tests["z_test"]["p_value_one_sided"] < alpha
diff_pp = stats["difference"]["absolute_pp"]
relative_pct = stats["difference"]["relative_pct"]

# ── Schema Banner ─────────────────────────────────────────────────
ctrl_label = stats["control"]["label"]
trt_label = stats["treatment"]["label"]
time_label = schema.time_col or "—"
id_label = schema.id_col or "—"
has_covariates = len(schema.covariate_cols) > 0
has_time = bool(schema.time_col)
has_id = bool(schema.id_col)

cov_impact = (
    "logistic regression (enriched model)" if has_covariates else "none selected"
) if True else "none selected"
time_impact = "temporal trends + enriched logistic regression" if has_time else "not used"
id_impact = "row identification only (not used in models)" if has_id else "not used"

st.markdown(f"""
<div class="schema-banner">
    <h3>📋 Analysis Configuration</h3>
    <div style="margin-bottom: 0.6rem;">
        <span class="schema-tag highlight">🎯 <strong>Target:</strong> {schema.target_col}</span>
        <span class="schema-tag highlight">🔀 <strong>Group:</strong> {schema.group_col}</span>
        <span class="schema-tag">📦 <strong>Covariates:</strong> {", ".join(schema.covariate_cols) if has_covariates else "—"}</span>
        <span class="schema-tag">🕐 <strong>Time:</strong> {time_label}</span>
        <span class="schema-tag">🆔 <strong>ID:</strong> {id_label}</span>
        <span class="schema-tag">📏 <strong>α:</strong> {alpha}</span>
        <span class="schema-tag"><strong>Control:</strong> "{ctrl_label}"</span>
        <span class="schema-tag"><strong>Treatment:</strong> "{trt_label}"</span>
    </div>
    <div style="font-size: 0.9rem; color: #475569; display: flex; gap: 1.5rem; flex-wrap: wrap;">
        <span>📦 <em>Covariates → {cov_impact}</em></span>
        <span>🕐 <em>Time → {time_impact}</em></span>
        <span>🆔 <em>ID → {id_impact}</em></span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Verdict Banner ─────────────────────────────────────────────────
if sig_two:
    if diff_pp > 0:
        verdict = "✅ DEPLOY — Statistically significant lift detected"
        verdict_color = "#059669"
    else:
        verdict = "❌ REJECT — Control is significantly better"
        verdict_color = "#dc2626"
elif bayes["p_treatment_better_pct"] > 95:
    verdict = "✅ DEPLOY — Bayesian probability > 95%"
    verdict_color = "#059669"
elif bayes["p_treatment_better_pct"] > 80 or tests["z_test"]["p_value_two_sided"] < 0.10:
    verdict = "⚠️ INCONCLUSIVE — Trend detected, extend the test"
    verdict_color = "#d97706"
else:
    verdict = "❌ No evidence of improvement — Keep control"
    verdict_color = "#64748b"

st.markdown(f"""
<div style="background: {verdict_color}; border-radius: 14px; padding: 1.2rem 2rem; margin-bottom: 1.5rem;">
    <p style="color: white; font-size: 1.25rem; font-weight: 700; margin: 0;">{verdict}</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────
kpi_cols = st.columns(5)
with kpi_cols[0]:
    st.metric(f"Control ({ctrl_label})", f"{stats['control']['rate_pct']:.2f}%", f"n = {stats['control']['n']:,}")
with kpi_cols[1]:
    st.metric(f"Treatment ({trt_label})", f"{stats['treatment']['rate_pct']:.2f}%", f"n = {stats['treatment']['n']:,}")
with kpi_cols[2]:
    st.metric("Difference", f"{diff_pp:+.3f} pp", f"{relative_pct:+.2f}% rel.", delta_color="inverse" if diff_pp < 0 else "normal")
with kpi_cols[3]:
    st.metric("P(T > C)", f"{bayes['p_treatment_better_pct']:.1f}%", "Bayesian")
with kpi_cols[4]:
    sig_tag = "✅ p < α" if sig_two else "❌ NS"
    st.metric("Z-test p-value", f"{tests['z_test']['p_value_two_sided']:.4f}", sig_tag)

# ── Detailed Results (Tabs) ───────────────────────────────────────
tabs = st.tabs(["📈 Overview", "🧪 Statistical Tests", "📐 Effect Sizes", "🔬 Bayesian", "📊 Segments", "⚡ Power & Robustness"])

# ── Tab 1: Overview ────────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-title'>Conversion Rates & Confidence Intervals</div>", unsafe_allow_html=True)
    vc1, vc2 = st.columns(2)

    with vc1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        grp_lbls = [ctrl_label, trt_label]
        rates_pct = [stats["control"]["rate_pct"], stats["treatment"]["rate_pct"]]
        colors = ["#6366f1", "#f59e0b"]
        bars = ax.bar(grp_lbls, rates_pct, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5, width=0.55)
        for bar, rate in zip(bars, rates_pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05 * max(rates_pct),
                    f"{rate:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11, color="#1e293b")
        ax.set_ylabel("Conversion Rate (%)", fontsize=10)
        ax.set_title("Conversion Rates by Group", fontweight="bold", fontsize=12, color="#1e293b")
        ax.set_ylim(0, max(rates_pct) * 1.3)
        sns.despine(left=True, bottom=True)
        ax.tick_params(colors="#64748b")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with vc2:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        y_pos = [0, 1]
        rates = [stats["control"]["rate"], stats["treatment"]["rate"]]
        ci_low = [stats["control"]["ci_95"][0], stats["treatment"]["ci_95"][0]]
        ci_high = [stats["control"]["ci_95"][1], stats["treatment"]["ci_95"][1]]
        for i, (r, lo, hi) in enumerate(zip(rates, ci_low, ci_high)):
            ax.errorbar(r, i, xerr=[[r - lo], [hi - r]], fmt="o", color=colors[i],
                        capsize=6, capthick=2, markersize=14, markeredgecolor="white", markeredgewidth=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(grp_lbls, fontsize=10)
        ax.set_xlabel("Conversion Rate", fontsize=10)
        ax.set_title("95% Confidence Intervals", fontweight="bold", fontsize=12, color="#1e293b")
        sns.despine(left=True, bottom=True)
        ax.tick_params(colors="#64748b")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Config Impact ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>Column Usage Summary</div>", unsafe_allow_html=True)
    usage_data = [
        ("🎯 Target", schema.target_col, "Binary outcome variable (0/1). Used in all statistical tests."),
        ("🔀 Group", schema.group_col, "Defines control vs. treatment. Used in all comparisons."),
    ]
    if has_covariates:
        for c in schema.covariate_cols:
            usage_data.append(("📦 Covariate", c, "Added to enriched logistic regression model to adjust for confounding."))
    else:
        usage_data.append(("📦 Covariates", "—", "None selected. Simple model only."))
    if has_time:
        usage_data.append(("🕐 Time", schema.time_col, "Used for daily trend visualization and as hour/weekend features in enriched logistic regression."))
    else:
        usage_data.append(("🕐 Time", "—", "Not provided. Temporal trends and time features skipped."))
    if has_id:
        usage_data.append(("🆔 ID", schema.id_col, "Recognized but not used in current statistical models."))
    else:
        usage_data.append(("🆔 ID", "—", "Not provided."))

    df_usage = pd.DataFrame(usage_data, columns=["Role", "Column", "Impact on Analysis"])
    st.dataframe(df_usage, hide_index=True, use_container_width=True)

    # ── Daily Trends ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>Daily Trends</div>", unsafe_allow_html=True)
    daily = R["temporal"].get("daily_data")
    if daily is not None and not daily.empty:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        for grp_val in daily["group"].unique():
            d = daily[daily["group"] == grp_val]
            lbl = ctrl_label if str(grp_val) == ctrl_label else trt_label
            color = colors[0] if lbl == ctrl_label else colors[1]
            ax.plot(d["date"], d["rate"] * 100, marker="o", label=lbl, color=color, linewidth=2, markersize=4)
        ax.set_ylabel("Conversion Rate (%)", fontsize=10)
        time_col_name = schema.time_col or "date"
        ax.set_title(f"Daily Conversion Rates by Group (from <{time_col_name}>)", fontweight="bold", fontsize=12, color="#1e293b")
        ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
        sns.despine(left=True, bottom=True)
        ax.tick_params(colors="#64748b")
        plt.xticks(rotation=30)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No time column provided or insufficient data for daily trends.", icon="🕐")

# ── Tab 2: Statistical Tests ──────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-title'>Hypothesis Tests</div>", unsafe_allow_html=True)

    test_rows = [
        ("Chi-squared", tests['chi_squared']['statistic'], tests['chi_squared']['p_value']),
        ("Z-test (two-sided)", tests['z_test']['z_statistic_two_sided'], tests['z_test']['p_value_two_sided']),
        ("Z-test (one-sided, T>C)", tests['z_test']['z_statistic_one_sided'], tests['z_test']['p_value_one_sided']),
        ("T-test (Welch)", tests['t_test']['t_statistic'], tests['t_test']['p_value']),
        ("Mann-Whitney U", tests['mann_whitney']['u_statistic'], tests['mann_whitney']['p_value']),
    ]
    df_t = pd.DataFrame(test_rows, columns=["Test", "Statistic", "p-value"])
    df_t["Sig. at α"] = df_t["p-value"].apply(lambda x: f"✅ {x:.4f} < {alpha}" if x < alpha else f"❌ {x:.4f} ≥ {alpha}")
    df_t["Statistic"] = df_t["Statistic"].apply(lambda x: f"{x:.4f}")
    df_t["p-value"] = df_t["p-value"].apply(lambda x: f"{x:.4f}")
    st.dataframe(df_t, hide_index=True, use_container_width=True)

    st.markdown("""
    <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem;">
        <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 10px; padding: 1rem 1.5rem; flex: 1;">
            <p style="margin: 0; font-size: 0.85rem; color: #059669; font-weight: 600;">Z-Test Interpretation</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.95rem; color: #065f46;">
            p-value {tests['z_test']['p_value_two_sided']:.4f} — 
            {"Statistically significant at α = " + str(alpha) if sig_two else "Not statistically significant at α = " + str(alpha)}.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tab 3: Effect Sizes ───────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-title'>Effect Size Estimates</div>", unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Cohen's h", f"{es['cohens_h']:.4f}", es['cohens_h_interpretation'])
    e2.metric("Odds Ratio", f"{es['odds_ratio']:.4f}", f"CI 95% [{es['odds_ratio_ci_95'][0]:.4f}, {es['odds_ratio_ci_95'][1]:.4f}]")
    e3.metric("Risk Ratio", f"{es['risk_ratio']:.4f}", "Relative risk")
    e4.metric("NNT", f"{es['nnt']:.1f}" if np.isfinite(es['nnt']) else "∞", "Number Needed to Treat")

    st.markdown("<div class='section-title'>Logistic Regression</div>", unsafe_allow_html=True)

    cov_list = ", ".join(schema.covariate_cols) if has_covariates else "none"
    time_note = f" + time features (hour, weekend)" if has_time else ""
    l1, l2 = st.columns(2)
    with l1:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem 1.5rem; height: 100%;">
            <h4 style="margin: 0 0 0.75rem 0; color: #1e293b;">Simple Model</h4>
            <p style="margin: 0.2rem 0; font-size: 0.85rem; color: #64748b;">Treatment only, no covariates</p>
            <p style="margin: 0.2rem 0;"><strong>OR:</strong> {R['log_simple']['odds_ratio']:.4f}</p>
            <p style="margin: 0.2rem 0;"><strong>p-value:</strong> {R['log_simple']['p_value']:.4f}</p>
            <p style="margin: 0.2rem 0;"><strong>CI 95%:</strong> [{R['log_simple']['or_ci_95'][0]:.4f}, {R['log_simple']['or_ci_95'][1]:.4f}]</p>
        </div>
        """, unsafe_allow_html=True)
    with l2:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem 1.5rem; height: 100%;">
            <h4 style="margin: 0 0 0.75rem 0; color: #1e293b;">Enriched Model</h4>
            <p style="margin: 0.2rem 0; font-size: 0.85rem; color: #64748b;">Covariates: {cov_list}{time_note}</p>
            <p style="margin: 0.2rem 0;"><strong>OR:</strong> {R['log_enriched']['odds_ratio']:.4f}</p>
            <p style="margin: 0.2rem 0;"><strong>p-value:</strong> {R['log_enriched']['p_value']:.4f}</p>
            <p style="margin: 0.2rem 0;"><strong>LR test vs. simple p:</strong> {R['lr_test']['p_value']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 4: Bayesian ───────────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-title'>Bayesian Beta-Binomial Analysis</div>", unsafe_allow_html=True)

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("P(Treatment > Control)", f"{bayes['p_treatment_better_pct']:.1f}%")
    bc2.metric("Expected Loss", f"{bayes['expected_loss']:.5f}")
    bc3.metric("CI 95% (diff)", f"[{bayes['ci_95'][0]*100:.2f}%, {bayes['ci_95'][1]*100:.2f}%]")
    dec_map = {"adopt_treatment": "✅ Adopt Treatment", "keep_control": "✅ Keep Control",
               "practical_equivalence": "↔️ Practical Equivalence", "insufficient_evidence": "❓ Need More Data"}
    bc4.metric("Decision", dec_map.get(bayes["decision"], bayes["decision"]))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    alphas_post = [1 + gs["conv_control"], 1 + gs["conv_treatment"]]
    betas_post = [1 + (gs["n_control"] - gs["conv_control"]), 1 + (gs["n_treatment"] - gs["conv_treatment"])]
    x_grid = np.linspace(0, max(stats["control"]["rate"] * 2, stats["treatment"]["rate"] * 2, 0.02), 300)
    from scipy.stats import beta as beta_dist
    ax.plot(x_grid, beta_dist.pdf(x_grid, alphas_post[0], betas_post[0]),
            label=f"Control ({ctrl_label})", color=colors[0], linewidth=2.5)
    ax.plot(x_grid, beta_dist.pdf(x_grid, alphas_post[1], betas_post[1]),
            label=f"Treatment ({trt_label})", color=colors[1], linewidth=2.5)
    ax.set_xlabel("Conversion Rate", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Posterior Distributions", fontweight="bold", fontsize=12, color="#1e293b")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    sns.despine(left=True, bottom=True)
    ax.tick_params(colors="#64748b")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Tab 5: Segments ───────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-title'>Segmentation Analysis</div>", unsafe_allow_html=True)
    if R["segmentation"]:
        seg_rows = []
        for col_name, segments in R["segmentation"].items():
            for seg_name, data in segments.items():
                if isinstance(data, dict) and "control_rate" in data:
                    seg_rows.append({
                        "Segment": f"{col_name}: {seg_name}",
                        "Control %": f"{data['control_rate']*100:.2f}%",
                        "Treatment %": f"{data['treatment_rate']*100:.2f}%",
                        "Diff pp": f"{(data['treatment_rate']-data['control_rate'])*100:+.2f}",
                        "p-value": f"{data['p_value']:.4f}",
                    })
        if seg_rows:
            st.dataframe(pd.DataFrame(seg_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No categorical segments found with sufficient data.", icon="📊")
    else:
        st.info("No segments available for analysis.", icon="📊")

# ── Tab 6: Power & Robustness ─────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-title'>Power Analysis</div>", unsafe_allow_html=True)
    pp1, pp2, pp3 = st.columns(3)
    pp1.metric("Observed Power", f"{power_obs*100:.1f}%")
    pp2.metric("MDE (Cohen's h)", f"{R.get('mde_h', 0):.4f}")
    pp3.metric("MDE (approx. pp)", f"±{R.get('mde_pp', 0)*100:.2f} pp")

    if power_obs * 100 < 80:
        n_str = f"{R.get('n_needed', float('inf')):,.0f}"
        st.warning(f"⚠️ Power is only {power_obs*100:.1f}% (below the 80% target). "
                   f"Need ~{n_str} observations per group to detect the observed effect size.", icon="⚡")

    st.markdown("<div class='section-title'>Robustness Checks</div>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        boot = R["robustness"]["bootstrap"]
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem 1.5rem; height: 100%;">
            <h4 style="margin: 0 0 0.75rem 0; color: #1e293b;">Bootstrap (non-parametric)</h4>
            <p style="margin: 0.2rem 0;"><strong>CI 95%:</strong> [{boot['ci_95'][0]*100:.2f}%, {boot['ci_95'][1]*100:.2f}%]</p>
            <p style="margin: 0.2rem 0;"><strong>CI 90%:</strong> [{boot['ci_90'][0]*100:.2f}%, {boot['ci_90'][1]*100:.2f}%]</p>
            <p style="margin: 0.2rem 0;"><strong>% positive:</strong> {boot['pct_positive']:.1f}%</p>
            <p style="margin: 0.2rem 0;"><strong>Mean diff:</strong> {boot['mean_diff']*100:.4f} pp</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        perm = R["robustness"]["permutation"]
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem 1.5rem; height: 100%;">
            <h4 style="margin: 0 0 0.75rem 0; color: #1e293b;">Permutation Test</h4>
            <p style="margin: 0.2rem 0;"><strong>p (two-sided):</strong> {perm['p_value_two_sided']:.4f}</p>
            <p style="margin: 0.2rem 0;"><strong>p (one-sided):</strong> {perm['p_value_one_sided']:.4f}</p>
            <p style="margin: 0.2rem 0;"><strong>Sig. (two-sided):</strong> {"✅ Yes" if perm['significant_two_sided'] else "❌ No"}</p>
            <p style="margin: 0.2rem 0;"><strong>Observed diff:</strong> {perm['observed_diff']*100:.4f} pp</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Flexible A/B Testing Analysis Toolkit · Professional Dashboard")
