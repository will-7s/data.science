import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from scipy.stats import beta as beta_dist

sns.set_theme(style="whitegrid")

from src.store import (
    dataset, col_meta, schema, all_cols, reset,
    target_candidates, group_candidates, datetime_cols,
    id_candidates, binary_cols, num_cols, cat_cols,
)
from src.data_loader import (
    load_data, resolve_column_names, prepare_data,
    get_group_stats, get_control_value,
)
from src.data_quality import report_data_quality
from src.descriptive_stats import compute_descriptive_stats
from src.hypothesis_testing import run_all_tests
from src.effect_size import compute_all_effect_sizes
from src.power_analysis import compute_power          # FIX: centralised module
from src.logistic_regression import (
    simple_logistic_regression,
    enriched_logistic_regression,
    likelihood_ratio_test,
)
from src.bayesian_analysis import beta_binomial_analysis
from src.segmentation import run_segmentation_analysis
from src.temporal_analysis import run_temporal_analysis
from src.robustness import run_robustness_checks
from src.report_generator import generate_text_report

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
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        border: none; color: white;
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4f46e5, #4338ca);
    }
    div[data-testid="metric-container"] {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: box-shadow 0.2s;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.8rem; color: #64748b; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.03em;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        font-size: 1.8rem; font-weight: 700; color: #0f172a;
    }
    .schema-banner {
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
        border: 1px solid #e0e7ff; border-radius: 16px;
        padding: 1.5rem 2rem; margin-bottom: 2rem;
    }
    .schema-banner h3 {
        margin: 0 0 0.75rem 0; color: #1e293b;
        font-size: 1.1rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .schema-tag {
        display: inline-block; background: white;
        border: 1px solid #cbd5e1; border-radius: 8px;
        padding: 0.4rem 0.8rem; margin: 0.2rem 0.4rem 0.2rem 0;
        font-size: 0.85rem;
    }
    .schema-tag strong { color: #4f46e5; }
    .schema-tag.highlight { border-color: #6366f1; background: #eef2ff; }
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1e293b;
        margin-top: 2rem; margin-bottom: 1rem;
        padding-bottom: 0.5rem; border-bottom: 2px solid #e2e8f0;
    }
    .stAlert { border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 0.5rem 1rem; font-weight: 500;
    }
    p, li { color: #334155; }
    hr { margin: 1.5rem 0; border-color: #e2e8f0; }
    .note-box {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 0.9rem 1.2rem;
        font-size: 0.85rem; color: #475569;
    }
    .cleaning-badge {
        display: inline-block; background: #fef3c7;
        border: 1px solid #fcd34d; border-radius: 6px;
        padding: 0.2rem 0.6rem; font-size: 0.8rem;
        color: #92400e; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
for key, default in [
    ("analysis_run", False),
    ("data_loaded",  False),
    ("prepared",     None),
    ("results",      {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center;margin-bottom:0;'>📊 A/B Testing</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#64748b;font-size:0.85rem;margin-top:0;'>"
        "Professional Analysis Toolkit</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Any tabular format with a binary target and a group column",
    )
    use_default = st.checkbox("Use default sample (ab_data.csv)", value=True)

    if st.button("📂 Load Data", type="secondary", width='stretch'):
        if uploaded_file:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
        elif use_default:
            path = str(Path(__file__).parent.parent / "ab_data.csv")
        else:
            path = None

        if path:
            with st.spinner("Loading data…"):
                df = load_data(Path(path))
                reset(df)
                st.session_state.data_loaded  = True
                st.session_state.analysis_run = False
                st.session_state.results      = {}
                st.toast(f"Loaded {len(df):,} rows × {len(df.columns)} columns", icon="✅")

    st.divider()

    if not st.session_state.data_loaded:
        st.info("👈 Load data first to configure column mapping")
        st.stop()

    st.markdown("### Column Mapping")
    auto_detect = st.checkbox("Auto-detect columns", value=True)

    default_target = (target_candidates or binary_cols or all_cols or [None])[0]
    default_group  = (group_candidates  or [None])[0]
    default_id     = (id_candidates     or [None])[0]
    default_time   = (datetime_cols     or [None])[0]

    idx_target = all_cols.index(default_target) if default_target in all_cols else 0
    idx_group  = all_cols.index(default_group)  if default_group  in all_cols else min(1, len(all_cols) - 1)

    selected_target = st.selectbox("🎯 Target (conversion)", all_cols, index=idx_target)
    selected_group  = st.selectbox("🔀 Group column",        all_cols, index=idx_group)

    # FIX: explicit control-group selector — no more alphabetical guessing
    group_values: list[str] = []
    if selected_group in dataset:
        group_values = sorted(str(v) for v in np.unique(dataset[selected_group]))
    selected_control = st.selectbox(
        "🏳️ Control group label",
        group_values,
        index=0,
        help="Which group value is the baseline / control?",
    )

    # Optional: landing-page column for mismatch cleaning
    page_col_opts = ["(None)"] + [c for c in all_cols if c not in (selected_target, selected_group)]
    selected_page_col = st.selectbox(
        "🗂️ Page/variant column (for mismatch cleaning)",
        page_col_opts,
        index=0,
        help="If group and page labels must match, select the page column to auto-remove mismatches",
    )

    covariate_opts      = [c for c in all_cols if c not in (selected_target, selected_group)]
    selected_covariates = st.multiselect("📦 Covariates (optional)", covariate_opts)

    time_opts = ["(None)"] + all_cols
    time_idx  = (all_cols.index(default_time) + 1) if (default_time and auto_detect and default_time in all_cols) else 0
    selected_time = st.selectbox("🕐 Time column (optional)", time_opts, index=time_idx)

    id_opts  = ["(None)"] + all_cols
    # FIX: exclude datetime columns from ID default
    id_idx = (all_cols.index(default_id) + 1) if (default_id and auto_detect and default_id in all_cols) else 0
    selected_id = st.selectbox("🆔 ID column (optional)", id_opts, index=id_idx)

    st.divider()
    alpha = st.slider("Significance level α", 0.01, 0.20, 0.05, 0.01, format="%.2f")

    run_btn = st.button("▶ Run Analysis", type="primary", width='stretch')

# ── Pre-run: Data Preview ─────────────────────────────────────────
if not st.session_state.analysis_run:
    st.markdown(
        "<h1 style='font-size:2rem;font-weight:800;color:#0f172a;'>"
        "📊 A/B Testing Analysis Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#64748b;font-size:1.05rem;margin-bottom:1.5rem;'>"
        "Upload any tabular dataset with a binary target and a group column.</p>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([2.2, 1])
    with col_left:
        st.markdown("#### Data Preview")
        preview_dict = {col: dataset[col][:10] for col in all_cols[:10]}
        st.dataframe(pd.DataFrame(preview_dict), width='stretch', hide_index=True)
    with col_right:
        st.markdown("#### Column Types")
        for c in all_cols:
            icon = "🔢" if col_meta.get(c) == "numeric" else "🏷️"
            st.markdown(f"{icon} **{c}** → {col_meta.get(c, '?')}")

    st.divider()
    st.markdown("#### Auto-detected Candidates")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Target",  len(target_candidates), ", ".join(target_candidates[:3]) or "—")
    c2.metric("🔀 Group",   len(group_candidates),  ", ".join(group_candidates[:3])  or "—")
    c3.metric("🕐 Time",    len(datetime_cols),      ", ".join(datetime_cols[:2])     or "—")
    c4.metric("🆔 ID",      len(id_candidates),      ", ".join(id_candidates[:2])     or "—")

    st.info("👈 Configure column mapping in the sidebar and click **▶ Run Analysis**", icon="💡")
    if not run_btn:
        st.stop()

# ── Run Analysis ──────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running A/B test analysis…"):
        page_col_arg = selected_page_col if selected_page_col != "(None)" else None

        resolve_column_names(
            target_col=selected_target,
            group_col=selected_group,
            covariate_cols=selected_covariates,
            time_col=selected_time if selected_time != "(None)" else "",
            id_col=selected_id   if selected_id   != "(None)" else "",
            control_value=selected_control,   # FIX: explicit control
        )

        prepared     = prepare_data(page_col=page_col_arg)   # FIX: cleaning runs here
        gs           = get_group_stats(prepared)
        stats        = compute_descriptive_stats(prepared)
        tests        = run_all_tests(prepared)
        effect_sizes = compute_all_effect_sizes(prepared)
        bayesian     = beta_binomial_analysis(prepared)

        # FIX: use centralised power_analysis module
        power = compute_power(
            effect_sizes["cohens_h"],
            gs["n_control"],
            gs["n_treatment"],
            alpha=alpha,
        )

        log_simple   = simple_logistic_regression(prepared)
        log_enriched = enriched_logistic_regression(prepared)
        lr_test      = likelihood_ratio_test(log_simple["model"], log_enriched["model"])
        segmentation = run_segmentation_analysis(prepared)
        temporal     = run_temporal_analysis(prepared)
        robustness   = run_robustness_checks(prepared)
        quality      = report_data_quality(cleaning_report=prepared.get("cleaning_report"))

        st.session_state.analysis_run = True
        st.session_state.prepared     = prepared
        st.session_state.results      = {
            "stats": stats, "tests": tests, "effect_sizes": effect_sizes,
            "bayesian": bayesian, "power": power,
            "log_simple": log_simple, "log_enriched": log_enriched,
            "lr_test": lr_test, "segmentation": segmentation,
            "temporal": temporal, "robustness": robustness,
            "quality": quality, "gs": gs, "alpha": alpha,
        }
    st.rerun()

# ── Load from session ─────────────────────────────────────────────
R            = st.session_state.results
stats        = R["stats"]
tests        = R["tests"]
es           = R["effect_sizes"]
bayes        = R["bayesian"]
power        = R["power"]
gs           = R["gs"]
alpha        = R["alpha"]
quality      = R["quality"]
cleaning     = quality.get("cleaning", {})

sig_two   = tests["z_test"]["p_value_two_sided"] < alpha
diff_pp   = stats["difference"]["absolute_pp"]
rel_pct   = stats["difference"]["relative_pct"]

# Global color palette (FIX: defined once at module scope, not inside a tab)
COLORS = ["#6366f1", "#f59e0b"]

ctrl_label = stats["control"]["label"]
trt_label  = stats["treatment"]["label"]
has_time   = bool(schema.time_col)
has_cov    = bool(schema.covariate_cols)
has_id     = bool(schema.id_col)

# ── Schema Banner ─────────────────────────────────────────────────
_cleaning_badge = ""
if cleaning:
    _cl_mm = f"{cleaning.get('n_mismatch_removed', 0):,}"
    _cl_dp = f"{cleaning.get('n_dupes_removed', 0):,}"
    _cl_cl = f"{cleaning.get('n_clean', '?'):,}"
    _cl_pc = f"{cleaning.get('pct_removed', 0):.2f}"
    _cleaning_badge = (
        f'<div class="cleaning-badge">🧹 {_cl_mm} mismatches + {_cl_dp} duplicates removed'
        f' — {_cl_cl} clean rows ({_cl_pc}% dropped)</div>'
    )
st.markdown(f"""
<div class="schema-banner">
    <h3>📋 Analysis Configuration</h3>
    <div style="margin-bottom: 0.6rem;">
        <span class="schema-tag highlight">🎯 <strong>Target:</strong> {schema.target_col}</span>
        <span class="schema-tag highlight">🔀 <strong>Group:</strong> {schema.group_col}</span>
        <span class="schema-tag highlight">🏳️ <strong>Control:</strong> &ldquo;{schema.control_value}&rdquo;</span>
        <span class="schema-tag">📦 <strong>Covariates:</strong> {", ".join(schema.covariate_cols) if has_cov else "—"}</span>
        <span class="schema-tag">🕐 <strong>Time:</strong> {schema.time_col or "—"}</span>
        <span class="schema-tag">🆔 <strong>ID:</strong> {schema.id_col or "—"}</span>
        <span class="schema-tag">📏 <strong>α:</strong> {alpha}</span>
    </div>
    {_cleaning_badge}
</div>
""", unsafe_allow_html=True)

# ── Verdict Banner ────────────────────────────────────────────────
if sig_two and diff_pp > 0:
    verdict, v_color = "✅ DEPLOY — Statistically significant lift detected", "#059669"
elif sig_two and diff_pp < 0:
    verdict, v_color = "❌ REJECT — Control is significantly better", "#dc2626"
elif bayes["p_treatment_better_pct"] > 95:
    verdict, v_color = "✅ DEPLOY — Bayesian probability > 95%", "#059669"
elif bayes["p_treatment_better_pct"] > 80 or tests["z_test"]["p_value_two_sided"] < 0.10:
    verdict, v_color = "⚠️ INCONCLUSIVE — Trend detected, extend the test", "#d97706"
else:
    verdict, v_color = "❌ No evidence of improvement — Keep control", "#64748b"

st.markdown(f"""
<div style="background:{v_color};border-radius:14px;padding:1.2rem 2rem;margin-bottom:1.5rem;">
    <p style="color:white;font-size:1.25rem;font-weight:700;margin:0;">{verdict}</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────
k0, k1, k2, k3, k4 = st.columns(5)
k0.metric(f"Control ({ctrl_label})",   f"{stats['control']['rate_pct']:.2f}%",   f"n = {stats['control']['n']:,}")
k1.metric(f"Treatment ({trt_label})",  f"{stats['treatment']['rate_pct']:.2f}%", f"n = {stats['treatment']['n']:,}")
k2.metric("Difference",
          f"{diff_pp:+.3f} pp",
          f"{rel_pct:+.2f}% rel.",
          delta_color="inverse" if diff_pp < 0 else "normal")
k3.metric("P(T > C)", f"{bayes['p_treatment_better_pct']:.1f}%", "Bayesian")
k4.metric("Z-test p-value",
          f"{tests['z_test']['p_value_two_sided']:.4f}",
          "✅ p < α" if sig_two else "❌ NS")

# ── Tabs ──────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Overview",
    "🧪 Statistical Tests",
    "📐 Effect Sizes",
    "🔬 Bayesian",
    "📊 Segments",
    "⚡ Power & Robustness",
    "📥 Export",
])

# ── Tab 0: Overview ───────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-title'>Conversion Rates & Confidence Intervals</div>",
                unsafe_allow_html=True)
    vc1, vc2 = st.columns(2)
    rates_pct = [stats["control"]["rate_pct"], stats["treatment"]["rate_pct"]]
    grp_lbls  = [ctrl_label, trt_label]

    with vc1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        bars = ax.bar(grp_lbls, rates_pct, color=COLORS, alpha=0.85,
                      edgecolor="white", linewidth=1.5, width=0.55)
        for bar, rate in zip(bars, rates_pct):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05 * max(rates_pct),
                    f"{rate:.2f}%", ha="center", va="bottom",
                    fontweight="bold", fontsize=11, color="#1e293b")
        ax.set_ylabel("Conversion Rate (%)", fontsize=10)
        ax.set_title("Conversion Rates by Group", fontweight="bold", fontsize=12, color="#1e293b")
        ax.set_ylim(0, max(rates_pct) * 1.3)
        sns.despine(left=True, bottom=True)
        ax.tick_params(colors="#64748b")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with vc2:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        rates = [stats["control"]["rate"], stats["treatment"]["rate"]]
        ci_lo = [stats["control"]["ci_95"][0],  stats["treatment"]["ci_95"][0]]
        ci_hi = [stats["control"]["ci_95"][1],  stats["treatment"]["ci_95"][1]]
        for i, (r, lo, hi) in enumerate(zip(rates, ci_lo, ci_hi)):
            ax.errorbar(r, i, xerr=[[r - lo], [hi - r]], fmt="o", color=COLORS[i],
                        capsize=6, capthick=2, markersize=14,
                        markeredgecolor="white", markeredgewidth=2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(grp_lbls, fontsize=10)
        ax.set_xlabel("Conversion Rate", fontsize=10)
        ax.set_title("95% Confidence Intervals (Wilson)", fontweight="bold",
                     fontsize=12, color="#1e293b")
        sns.despine(left=True, bottom=True)
        ax.tick_params(colors="#64748b")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    # Daily Trends
    st.markdown("<div class='section-title'>Daily Trends</div>", unsafe_allow_html=True)
    daily = R["temporal"].get("daily_data")
    if daily is not None and not daily.empty:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        for grp_val in daily["group"].unique():
            d   = daily[daily["group"] == grp_val]
            lbl = ctrl_label if str(grp_val) == ctrl_label else trt_label
            col = COLORS[0] if lbl == ctrl_label else COLORS[1]
            ax.plot(d["date"], d["rate"] * 100, marker="o", label=lbl,
                    color=col, linewidth=2, markersize=4)
        ax.set_ylabel("Conversion Rate (%)", fontsize=10)
        ax.set_title(f"Daily Conversion Rates (from '{schema.time_col}')",
                     fontweight="bold", fontsize=12, color="#1e293b")
        ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
        sns.despine(left=True, bottom=True)
        ax.tick_params(colors="#64748b")
        plt.xticks(rotation=30)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        # Trend significance note
        trends = R["temporal"].get("trends", {})
        if trends:
            trend_lines = []
            for g, t in trends.items():
                if not np.isnan(t.get("pearson_r", float("nan"))):
                    trend_lines.append(
                        f"**{g}**: r = {t['pearson_r']:.3f}, p = {t['p_value']:.4f}"
                        f" {'✅ significant trend' if t['significant_trend'] else '(no trend)'}"
                    )
            if trend_lines:
                st.markdown(
                    "<div class='note-box'>⚖️ <strong>Weighted trend test</strong> "
                    "(Pearson r weighted by √n per day): "
                    + " &nbsp;|&nbsp; ".join(trend_lines) + "</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No time column provided or insufficient data for daily trends.", icon="🕐")

    # Column usage summary
    st.markdown("<div class='section-title'>Column Usage Summary</div>", unsafe_allow_html=True)
    usage = [
        ("🎯 Target",     schema.target_col, "Binary outcome. Used in all tests."),
        ("🔀 Group",      schema.group_col,  "Defines control vs. treatment."),
        ("🏳️ Control",   schema.control_value, "Baseline group label (explicit selection)."),
    ]
    if has_cov:
        for c in schema.covariate_cols:
            usage.append(("📦 Covariate", c, "Added to enriched logistic regression."))
    else:
        usage.append(("📦 Covariates", "—", "None selected — simple model only."))
    usage.append(("🕐 Time", schema.time_col or "—",
                  "Daily trends + time features in enriched model." if has_time else "Not provided."))
    usage.append(("🆔 ID", schema.id_col or "—",
                  "Used for deduplication in data cleaning." if has_id else "Not provided."))
    st.dataframe(pd.DataFrame(usage, columns=["Role", "Column", "Impact"]),
                 hide_index=True, width='stretch')

# ── Tab 1: Statistical Tests ──────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-title'>Hypothesis Tests</div>", unsafe_allow_html=True)

    test_rows = [
        ("Chi-squared",           tests["chi_squared"]["statistic"],           tests["chi_squared"]["p_value"]),
        ("Z-test (two-sided)",    tests["z_test"]["z_statistic_two_sided"],     tests["z_test"]["p_value_two_sided"]),
        ("Z-test (one-sided T>C)",tests["z_test"]["z_statistic_one_sided"],     tests["z_test"]["p_value_one_sided"]),
        ("T-test (Welch)",        tests["t_test"]["t_statistic"],               tests["t_test"]["p_value"]),
        ("Mann-Whitney U",        tests["mann_whitney"]["u_statistic"],         tests["mann_whitney"]["p_value"]),
    ]
    df_t = pd.DataFrame(test_rows, columns=["Test", "Statistic", "p-value"])
    df_t["Sig. at α"] = df_t["p-value"].apply(
        lambda x: f"✅ {x:.4f} < {alpha}" if x < alpha else f"❌ {x:.4f} ≥ {alpha}"
    )
    df_t["Statistic"] = df_t["Statistic"].apply(lambda x: f"{x:.4f}")
    df_t["p-value"]   = df_t["p-value"].apply(lambda x: f"{x:.4f}")
    st.dataframe(df_t, hide_index=True, width='stretch')

    # FIX: was missing f-prefix — values were shown as raw Python code
    p_two_val = tests["z_test"]["p_value_two_sided"]
    p_one_val = tests["z_test"]["p_value_one_sided"]
    interp_two = ("Statistically significant" if sig_two else "Not statistically significant") + f" at α = {alpha}"
    interp_one = ("Treatment > Control" if tests["z_test"]["significant_one_sided"] else "No directional evidence") + f" at α = {alpha}"

    st.markdown(f"""
    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:1rem;">
        <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;
                    padding:1rem 1.5rem;flex:1;">
            <p style="margin:0;font-size:0.85rem;color:#059669;font-weight:600;">
                Z-Test (two-sided)</p>
            <p style="margin:0.3rem 0 0;font-size:0.95rem;color:#065f46;">
                p = {p_two_val:.4f} — {interp_two}</p>
        </div>
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                    padding:1rem 1.5rem;flex:1;">
            <p style="margin:0;font-size:0.85rem;color:#1d4ed8;font-weight:600;">
                Z-Test one-sided (H₁: T &gt; C)</p>
            <p style="margin:0.3rem 0 0;font-size:0.95rem;color:#1e3a8a;">
                p = {p_one_val:.4f} — {interp_one}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='note-box' style='margin-top:1rem;'>
    📌 <strong>One-sided test</strong> correctly tests H₁: p_treatment &gt; p_control.
    Reject H₀ only if you are willing to commit to this direction <em>before</em> seeing the data.
    </div>
    """, unsafe_allow_html=True)

# ── Tab 2: Effect Sizes ───────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-title'>Effect Size Estimates</div>", unsafe_allow_html=True)

    nnt_label = es.get("nnt_label", "NNT")
    nnt_dir   = es.get("nnt_direction", "")
    nnt_val   = es["nnt"]

    e0, e1, e2, e3 = st.columns(4)
    e0.metric("Cohen's h", f"{es['cohens_h']:.4f}", es["cohens_h_interpretation"])
    e1.metric("Odds Ratio", f"{es['odds_ratio']:.4f}",
              f"CI 95% [{es['odds_ratio_ci_95'][0]:.3f}, {es['odds_ratio_ci_95'][1]:.3f}]")
    e2.metric("Risk Ratio", f"{es['risk_ratio']:.4f}", "Relative risk T/C")
    # FIX: show NNT or NNH correctly
    e3.metric(
        nnt_label,
        f"{nnt_val:.1f}" if np.isfinite(nnt_val) else "∞",
        f"Number Needed to {'Treat' if nnt_label == 'NNT' else 'Harm'} ({nnt_dir})",
    )

    st.markdown("<div class='section-title'>Logistic Regression</div>", unsafe_allow_html=True)
    cov_note  = ", ".join(schema.covariate_cols) if has_cov else "none"
    time_note = " + hour/weekend" if has_time else ""

    l0, l1 = st.columns(2)
    with l0:
        ls = R["log_simple"]
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                    padding:1.2rem 1.5rem;">
            <h4 style="margin:0 0 0.75rem;color:#1e293b;">Simple Model</h4>
            <p style="margin:0.2rem 0;font-size:0.85rem;color:#64748b;">
                Treatment only, no covariates</p>
            <p style="margin:0.2rem 0;"><strong>OR:</strong> {ls['odds_ratio']:.4f}</p>
            <p style="margin:0.2rem 0;"><strong>CI 95%:</strong>
                [{ls['or_ci_95'][0]:.4f}, {ls['or_ci_95'][1]:.4f}]</p>
            <p style="margin:0.2rem 0;"><strong>p-value:</strong> {ls['p_value']:.4f}
                {"✅" if ls['significant'] else "❌"}</p>
        </div>
        """, unsafe_allow_html=True)
    with l1:
        le = R["log_enriched"]
        lr = R["lr_test"]
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                    padding:1.2rem 1.5rem;">
            <h4 style="margin:0 0 0.75rem;color:#1e293b;">Enriched Model</h4>
            <p style="margin:0.2rem 0;font-size:0.85rem;color:#64748b;">
                Covariates: {cov_note}{time_note}</p>
            <p style="margin:0.2rem 0;"><strong>OR:</strong> {le['odds_ratio']:.4f}</p>
            <p style="margin:0.2rem 0;"><strong>CI 95%:</strong>
                [{le['or_ci_95'][0]:.4f}, {le['or_ci_95'][1]:.4f}]</p>
            <p style="margin:0.2rem 0;"><strong>p-value:</strong> {le['p_value']:.4f}
                {"✅" if le['significant'] else "❌"}</p>
            <p style="margin:0.2rem 0;"><strong>LR test vs. simple:</strong>
                LR = {lr['lr_statistic']:.2f}, p = {lr['p_value']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 3: Bayesian ───────────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-title'>Bayesian Beta-Binomial Analysis</div>",
                unsafe_allow_html=True)

    bc0, bc1, bc2, bc3 = st.columns(4)
    bc0.metric("P(Treatment > Control)", f"{bayes['p_treatment_better_pct']:.1f}%")
    bc1.metric("Expected Loss",          f"{bayes['expected_loss']:.5f}")
    rope_lo = bayes.get("rope_lower", -0.002)
    rope_hi = bayes.get("rope_upper",  0.002)
    bc2.metric("P(diff in ROPE)",
               f"{bayes['p_rope_region']*100:.1f}%",
               f"ROPE [{rope_lo:+.3f}, {rope_hi:+.3f}]")
    dec_map = {
        "adopt_treatment":       "✅ Adopt Treatment",
        "keep_control":          "✅ Keep Control",
        "practical_equivalence": "↔️ Practical Equiv.",
        "insufficient_evidence": "❓ Need More Data",
    }
    bc3.metric("Decision", dec_map.get(bayes["decision"], bayes["decision"]))

    # Posterior distributions — COLORS is global so no NameError
    fig, ax = plt.subplots(figsize=(9, 3.8))
    a_c = 1 + gs["conv_control"]
    b_c = 1 + (gs["n_control"]   - gs["conv_control"])
    a_t = 1 + gs["conv_treatment"]
    b_t = 1 + (gs["n_treatment"]  - gs["conv_treatment"])
    r_max = max(stats["control"]["rate"], stats["treatment"]["rate"]) * 2
    x_grid = np.linspace(0, max(r_max, 0.02), 400)
    ax.plot(x_grid, beta_dist.pdf(x_grid, a_c, b_c),
            label=f"Control ({ctrl_label})",   color=COLORS[0], linewidth=2.5)
    ax.plot(x_grid, beta_dist.pdf(x_grid, a_t, b_t),
            label=f"Treatment ({trt_label})", color=COLORS[1], linewidth=2.5)
    ax.fill_between(x_grid, beta_dist.pdf(x_grid, a_c, b_c),
                    alpha=0.12, color=COLORS[0])
    ax.fill_between(x_grid, beta_dist.pdf(x_grid, a_t, b_t),
                    alpha=0.12, color=COLORS[1])
    ax.set_xlabel("Conversion Rate", fontsize=10)
    ax.set_ylabel("Posterior Density", fontsize=10)
    ax.set_title("Posterior Distributions (Beta-Binomial)",
                 fontweight="bold", fontsize=12, color="#1e293b")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    sns.despine(left=True, bottom=True)
    ax.tick_params(colors="#64748b")
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    n_sims = bayes.get("n_simulations", 50_000)
    st.markdown(
        f"<div class='note-box'>Prior: Beta(1, 1) — uniform. "
        f"Simulations: {n_sims:,}. "
        f"ROPE: [{rope_lo:+.3f}, {rope_hi:+.3f}] (configurable in config.py). "
        f"CI 95%: [{bayes['ci_95'][0]*100:.2f}%, {bayes['ci_95'][1]*100:.2f}%].</div>",
        unsafe_allow_html=True,
    )

# ── Tab 4: Segments ───────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-title'>Segmentation Analysis</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='note-box'>p-values corrected for multiple comparisons "
        "using Benjamini-Hochberg FDR. "
        "✅ = significant after correction.</div>",
        unsafe_allow_html=True,
    )

    seg_rows = []
    for col_name, segments in R["segmentation"].items():
        for seg_name, data in segments.items():
            if not (isinstance(data, dict) and "control_rate" in data):
                continue
            diff_seg = (data["treatment_rate"] - data["control_rate"]) * 100
            seg_rows.append({
                "Segment":      f"{col_name}: {seg_name}",
                "Control %":    f"{data['control_rate']*100:.2f}%",
                "n Control":    data["control_n"],
                "Treatment %":  f"{data['treatment_rate']*100:.2f}%",
                "n Treatment":  data["treatment_n"],
                "Diff (pp)":    f"{diff_seg:+.2f}",
                "p (raw)":      f"{data.get('p_value_raw', data.get('p_value', 1.0)):.4f}",
                "p (FDR adj.)": f"{data.get('p_value', 1.0):.4f}",
                "Sig.*":        "✅" if data.get("significant") else "❌",
            })

    if seg_rows:
        st.dataframe(pd.DataFrame(seg_rows), hide_index=True, width='stretch')

        # Segment bar chart
        if len(seg_rows) > 1:
            fig, ax = plt.subplots(figsize=(max(8, len(seg_rows) * 1.4), 4))
            df_seg  = pd.DataFrame(seg_rows)
            x = np.arange(len(df_seg))
            w = 0.35
            ctrl_vals = [float(v.rstrip("%")) for v in df_seg["Control %"]]
            trt_vals  = [float(v.rstrip("%")) for v in df_seg["Treatment %"]]
            ax.bar(x - w / 2, ctrl_vals, w, label="Control",   color=COLORS[0], alpha=0.85)
            ax.bar(x + w / 2, trt_vals,  w, label="Treatment", color=COLORS[1], alpha=0.85)
            for i, row in df_seg.iterrows():
                diff_v = float(row["Diff (pp)"])
                col_d  = "#059669" if diff_v > 0 else "#dc2626"
                ax.annotate(f"{diff_v:+.2f}pp",
                            (i, max(ctrl_vals[i], trt_vals[i]) + 0.15),
                            ha="center", fontsize=8, color=col_d, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(df_seg["Segment"], rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Conversion Rate (%)", fontsize=10)
            ax.set_title("Conversion Rate by Segment", fontweight="bold",
                         fontsize=12, color="#1e293b")
            ax.legend(fontsize=10)
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close(fig)
    else:
        st.info("No categorical segments found with sufficient data.", icon="📊")

# ── Tab 5: Power & Robustness ─────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-title'>Power Analysis</div>", unsafe_allow_html=True)

    if power["skipped"]:
        st.info("Power analysis skipped — effect size is effectively zero.", icon="⚡")
    else:
        pp0, pp1, pp2, pp3 = st.columns(4)
        pp0.metric("Observed Power",  f"{power['power_observed']*100:.1f}%")
        pp1.metric("N/group for 80%", f"{power['n_needed_80pct']:,.0f}")
        pp2.metric("MDE (Cohen's h)", f"{power['mde_cohens_h']:.4f}")
        pp3.metric("MDE (approx.)",   f"±{power['mde_pp']*100:.2f} pp")

        if power["power_observed"] < 0.80:
            st.warning(
                f"⚠️ Observed power is only {power['power_observed']*100:.1f}% (target: 80%). "
                f"You need ~{power['n_needed_80pct']:,.0f} observations per group to detect "
                f"this effect size reliably.",
                icon="⚡",
            )

        # Power curve
        fig, ax = plt.subplots(figsize=(7, 3.5))
        from src.power_analysis import compute_power as _cp
        ns = np.linspace(1000, max(int(power["n_needed_80pct"] * 1.5), gs["n_control"] * 2), 80).astype(int)
        pws = [_cp(es["cohens_h"], int(n), int(n), alpha)["power_observed"] for n in ns]
        ax.plot(ns, [p * 100 for p in pws], color=COLORS[0], linewidth=2)
        ax.axhline(80, color="#dc2626", linestyle="--", linewidth=1.2, label="80% target")
        ax.axvline(gs["n_control"], color=COLORS[1], linestyle=":", linewidth=1.2,
                   label=f"Current n={gs['n_control']:,}")
        ax.set_xlabel("Sample size per group", fontsize=10)
        ax.set_ylabel("Statistical Power (%)", fontsize=10)
        ax.set_title("Power vs. Sample Size", fontweight="bold", fontsize=12, color="#1e293b")
        ax.legend(fontsize=9)
        sns.despine(left=True, bottom=True)
        ax.tick_params(colors="#64748b")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    st.markdown("<div class='section-title'>Robustness Checks</div>", unsafe_allow_html=True)
    r0, r1 = st.columns(2)
    boot = R["robustness"]["bootstrap"]
    perm = R["robustness"]["permutation"]

    with r0:
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:1.2rem 1.5rem;">
            <h4 style="margin:0 0 0.75rem;color:#1e293b;">Bootstrap (non-parametric)</h4>
            <p style="margin:0.2rem 0;"><strong>CI 95%:</strong>
                [{boot['ci_95'][0]*100:.3f}%, {boot['ci_95'][1]*100:.3f}%]</p>
            <p style="margin:0.2rem 0;"><strong>CI 90%:</strong>
                [{boot['ci_90'][0]*100:.3f}%, {boot['ci_90'][1]*100:.3f}%]</p>
            <p style="margin:0.2rem 0;"><strong>% samples T &gt; C:</strong>
                {boot['pct_positive']:.1f}%</p>
            <p style="margin:0.2rem 0;"><strong>Mean diff:</strong>
                {boot['mean_diff']*100:.4f} pp</p>
        </div>
        """, unsafe_allow_html=True)

    with r1:
        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:1.2rem 1.5rem;">
            <h4 style="margin:0 0 0.75rem;color:#1e293b;">Permutation Test</h4>
            <p style="margin:0.2rem 0;"><strong>p (two-sided):</strong>
                {perm['p_value_two_sided']:.4f}
                {"✅" if perm['significant_two_sided'] else "❌"}</p>
            <p style="margin:0.2rem 0;"><strong>p (one-sided):</strong>
                {perm['p_value_one_sided']:.4f}
                {"✅" if perm['significant_one_sided'] else "❌"}</p>
            <p style="margin:0.2rem 0;"><strong>Observed diff:</strong>
                {perm['observed_diff']*100:.4f} pp</p>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 6: Export ─────────────────────────────────────────────────
with tabs[6]:
    st.markdown("<div class='section-title'>Export Results</div>", unsafe_allow_html=True)

    # FIX: generate full text report and offer download — was completely absent
    try:
        from src.power_analysis import compute_power as _cp2
        report_txt = generate_text_report(
            quality=R["quality"],
            stats=stats,
            tests=tests,
            effect_sizes=es,
            power=power,
            bayesian=bayes,
            segmentation=R["segmentation"],
            temporal=R["temporal"],
            robustness=R["robustness"],
            log_reg_simple=R["log_simple"],
            log_reg_enriched=R["log_enriched"],
            lr_test=R["lr_test"],
        )
        st.download_button(
            label="📄 Download Full Report (.txt)",
            data=report_txt.encode("utf-8"),
            file_name="ab_testing_report.txt",
            mime="text/plain",
            width='stretch',
        )
        with st.expander("Preview report"):
            st.code(report_txt, language=None)
    except Exception as exc:
        st.error(f"Could not generate report: {exc}")

    # CSV export of key metrics
    st.markdown("---")
    summary_rows = [
        ("Control n",              gs["n_control"]),
        ("Treatment n",            gs["n_treatment"]),
        ("Control rate (%)",       round(stats["control"]["rate_pct"], 4)),
        ("Treatment rate (%)",     round(stats["treatment"]["rate_pct"], 4)),
        ("Absolute diff (pp)",     round(diff_pp, 4)),
        ("Relative diff (%)",      round(rel_pct, 4)),
        ("Z-test p (two-sided)",   round(tests["z_test"]["p_value_two_sided"], 6)),
        ("Z-test p (one-sided)",   round(tests["z_test"]["p_value_one_sided"], 6)),
        ("Cohen's h",              round(es["cohens_h"], 6)),
        ("Odds Ratio",             round(es["odds_ratio"], 6)),
        ("P(T>C) Bayesian (%)",    round(bayes["p_treatment_better_pct"], 2)),
        ("Bayesian decision",      bayes["decision"]),
        ("Observed power (%)",     round(power.get("power_observed", 0) * 100, 2)),
        ("Bootstrap CI 95% low",   round(R["robustness"]["bootstrap"]["ci_95"][0] * 100, 4)),
        ("Bootstrap CI 95% high",  round(R["robustness"]["bootstrap"]["ci_95"][1] * 100, 4)),
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    df_summary["Value"] = df_summary["Value"].astype(str)
    st.download_button(
        label="📊 Download Summary Metrics (.csv)",
        data=df_summary.to_csv(index=False).encode("utf-8"),
        file_name="ab_test_summary.csv",
        mime="text/csv",
        width='stretch',
    )
    st.dataframe(df_summary, hide_index=True, width='stretch')

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "A/B Testing Analysis Toolkit · "
    "Corrections: data cleaning · one-sided Z-test direction · "
    "logistic regression OR · FDR segmentation · "
    "ROPE calibration · bootstrap sample size · export"
)
