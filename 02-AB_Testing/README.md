# A/B Testing Analysis Project

A comprehensive A/B testing analysis toolkit for evaluating conversion rate experiments.

## Features

- **Data Quality Checks** — Detects inconsistencies, duplicate users, and misassignments
- **Descriptive Statistics** — Conversion rates, confidence intervals, group comparisons
- **Frequentist Hypothesis Testing** — Chi-squared, Z-test, T-test, Mann-Whitney U
- **Effect Size Metrics** — Cohen's h, Odds Ratio, Risk Ratio, NNT, Phi coefficient
- **Logistic Regression** — Simple, enriched (hour + weekend), and interaction models
- **Bayesian A/B Testing** — Beta-Binomial model with ROPE decision framework
- **Segmentation Analysis** — By hour of day, weekday/weekend with Bonferroni correction
- **Temporal Analysis** — Daily trends and correlation tests
- **Robustness Checks** — Bootstrap confidence intervals and permutation tests
- **Statistical Power** — Post-hoc power, minimum detectable effect, required sample size
- **Interactive Dashboard** — Streamlit web app for visual exploration
- **Report Generation** — Text-based comprehensive report

## Project Structure

```
.
├── ab_data.csv              # Raw dataset
├── config.py                # Global configuration
├── run.py                   # CLI entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load, clean, prepare data
│   ├── data_quality.py      # Quality metrics report
│   ├── descriptive_stats.py # Conversion rates, CIs
│   ├── hypothesis_testing.py# Chi2, Z-test, T-test, MWU
│   ├── effect_size.py       # Cohen's h, OR, RR, NNT, Phi
│   ├── logistic_regression.py# GLM models
│   ├── bayesian_analysis.py # Beta-Binomial analysis
│   ├── segmentation.py      # Segment analysis
│   ├── temporal_analysis.py # Time series analysis
│   ├── robustness.py        # Bootstrap & permutation
│   ├── visualizations.py    # Matplotlib/seaborn charts
│   └── report_generator.py  # Text report generation
├── app/
│   ├── __init__.py
│   └── dashboard.py         # Streamlit dashboard
├── outputs/
│   ├── reports/             # Generated reports
│   └── plots/               # Generated visualizations
└── notebooks/
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Analysis (CLI)

```bash
python run.py
```

Options:
- `--data path/to/data.csv` — Specify custom dataset
- `--output path/to/output` — Custom output directory
- `--no-plots` — Skip plot generation

### 3. Launch Interactive Dashboard

```bash
streamlit run app/dashboard.py
```

## Dataset

The dataset contains **294,478 rows** from a classic A/B test:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `timestamp` | Event timestamp |
| `group` | `control` (old page) or `treatment` (new page) |
| `landing_page` | `old_page` or `new_page` |
| `converted` | Binary conversion flag (0/1) |

## Key Findings (from example dataset)

- **Control conversion rate:** 12.04%
- **Treatment conversion rate:** 11.88%
- **Difference:** -0.16 pp (treatment performs *worse*)
- **P(treatment > control):** 9.5% (Bayesian)
- **Conclusion:** No statistically significant difference — **do not deploy** the new page.

## License

MIT
