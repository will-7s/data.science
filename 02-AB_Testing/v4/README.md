# A/B Testing Analysis App

Interactive A/B testing analysis platform combining frequentist and Bayesian methods with real-time visualizations.

## Features

- **Data Loading** — Upload CSV, Excel, Parquet or JSON files with automatic column detection
- **Data Quality** — Automatic mismatch and duplicate detection with cleaning report
- **Descriptive Statistics** — Conversion rates, Wilson confidence intervals, group comparisons
- **Frequentist Tests** — Z-test, T-test with configurable significance level (α)
- **Bayesian Analysis** — Beta-Binomial (binary) and Normal-Normal (continuous) models with ROPE decision framework
- **Effect Sizes** — Cohen's d/h, Odds Ratio, Risk Ratio, NNT, Phi coefficient, lift metrics
- **Logistic Regression** — Simple, enriched (covariates), and likelihood-ratio tests
- **Segmentation Analysis** — Automatic segment discovery with Bonferroni correction
- **Temporal Analysis** — Daily conversion trends with statistical correlation
- **Robustness Checks** — Bootstrap confidence intervals and permutation tests
- **Power Analysis** — Post-hoc power, minimum detectable effect, required sample size
- **Confidence Scoring** — Composite score combining power, Bayesian probability, and CI width
- **SPRT Monitoring** — Sequential Probability Ratio Test for ongoing experiments
- **Interactive Charts** — Conversion bars, forest plots, posterior distributions, power curves, segment trees, trend lines
- **Theme System** — Light/dark mode with system preference detection
- **Export** — Text report and CSV summary download

## Project Structure

```
.
├── app/                        # Dash web application
│   ├── callbacks/              # Analysis, export, tab callbacks
│   ├── charts/                 # Plotly chart modules
│   ├── components/             # Reusable UI components
│   ├── layout.py               # Main application layout
│   ├── theme.py                # Light/dark theme system
│   └── export_utils.py         # Report generation utilities
├── assets/
│   └── style.css               # Design token system v2
├── src/                        # Analysis engine
│   ├── data_loader.py          # Load, clean, prepare data
│   ├── data_cache.py           # In-memory cache with LRU eviction
│   ├── hypothesis_testing.py   # Z-test, T-test implementations
│   ├── bayesian_analysis.py    # Beta-Binomial & Normal-Normal
│   ├── effect_size.py          # Cohen's d/h, OR, RR, NNT
│   ├── logistic_regression.py  # GLM with covariates
│   ├── power_analysis.py       # Statistical power computations
│   ├── robustness.py           # Bootstrap & permutation tests
│   ├── segmentation.py         # Segment discovery engine
│   ├── temporal_analysis.py    # Time series analysis
│   ├── sprt.py                 # Sequential testing
│   ├── confidence_score.py     # Composite scoring
│   ├── report_generator.py     # JSON report export
│   ├── descriptive_stats.py    # Summary statistics
│   └── compute.py              # Async compute utilities
├── config.py                   # Global configuration
├── run.py                      # Application entry point
├── wsgi.py                     # WSGI server (Gunicorn)
├── Dockerfile                  # Containerized deployment
├── requirements-prod.txt       # Production dependencies
└── README.md
```

## Quick Start

### Local Development

```bash
pip install -r requirements.txt
python run.py --dev
```

### Production (Docker)

```bash
docker build -t ab-testing-app .
docker run -p 7860:7860 ab-testing-app
```

### CLI Export

```bash
python run.py data.csv --export-json report.json
```

## Online Demo

Deployed on Hugging Face Spaces:

- **App:** https://will-7s-ab-test-app.hf.space
- **Space:** https://huggingface.co/spaces/will-7s/ab_test_app

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_level` | 0.95 | Statistical confidence level |
| `bootstrap_iterations` | 1,000 | Bootstrap resampling count |
| `permutation_iterations` | 1,000 | Permutation test iterations |
| `bayesian_simulations` | 10,000 | MCMC simulation count |
| `sprt.alpha` | 0.05 | SPRT type I error rate |
| `sprt.beta` | 0.20 | SPRT type II error rate |

## Tech Stack

- **Frontend:** Dash 4.x, Plotly 6.x
- **Backend:** Flask, Gunicorn
- **Analysis:** NumPy, Pandas, SciPy, Statsmodels
- **Deployment:** Docker, Hugging Face Spaces
