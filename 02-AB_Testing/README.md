# 📊 A/B Testing Analysis Project

A/B testing toolkit for evaluating conversion rate experiments, combining frequentist and Bayesian methods. Includes an interactive Dash web app with real-time visualizations.

To test the app: https://will-7s-ab-test-app.hf.space

---

## 🔎 Overview

A/B testing is a core methodology for data-driven decision making. This project provides a complete analysis pipeline to compare two variants (control vs treatment) and determine whether an observed difference is statistically significant.

The analysis combines multiple approaches to build a robust verdict:

- **Frequentist tests** — Z-test, T-test with configurable significance thresholds
- **Bayesian analysis** — Beta-Binomial and Normal-Normal models with ROPE decision framework
- **Robustness checks** — Bootstrap confidence intervals and permutation tests
- **Effect sizes** — Cohen's d/h, Odds Ratio, Risk Ratio, NNT
- **Power analysis** — Post-hoc power, minimum detectable effect

---

## 🎯 Features

### v4 — Dash Web Application

Current version with an interactive Dash interface:

- **Data Loading** — Upload CSV, Excel, Parquet or JSON files
- **Statistical Tests** — Frequentist + Bayesian hybrid decision engine
- **Interactive Charts** — Conversion bars, posterior distributions, power curves, trend lines
- **Segmentation** — Automatic segment discovery with Bonferroni correction
- **Temporal Analysis** — Daily conversion trends and correlation tests
- **SPRT Monitoring** — Sequential Probability Ratio Test
- **Export** — Text reports and CSV summaries
- **Theme System** — Light/dark mode

### v1–v3 — Earlier Prototypes

Progressive versions building from a Streamlit dashboard (v1) to the full Dash application (v4), adding Bayesian analysis, power calculations, logistic regression, and robustness checks across releases.

---

## 🧠 Methodology

The analysis follows a structured workflow:

1. **Load & Validate**  
   Upload data, detect column types, identify target and group columns.

2. **Prepare**  
   Filter control/treatment groups, handle covariates and time columns.

3. **Analyze**  
   Run parallel statistical tests: frequentist, Bayesian, bootstrap, permutation, segmentation, and temporal analysis.

4. **Score & Decide**  
   Compute a composite confidence score combining power, Bayesian probability, and confidence interval width. Deliver a verdict: Deploy, Reject, Inconclusive, or No Evidence.

---

## 🛠️ Tech Stack

- **Frontend:** Dash 4.x, Plotly 6.x
- **Backend:** Flask, Gunicorn
- **Analysis:** NumPy, Pandas, SciPy, Statsmodels
- **Deployment:** Docker, Hugging Face Spaces
- **Testing:** pytest, Hypothesis (property-based)

---

## ❓ Key Questions

- Is the treatment significantly better than the control?
- What is the probability that the treatment outperforms the control?
- How large is the effect? Is it practically significant?
- Is the result robust to resampling and permutations?
- Do certain segments respond differently to the treatment?
- How much statistical power does the experiment have?
- Should we deploy the change, extend the test, or reject?

---

## 📁 Project Structure

```
v1/    Streamlit dashboard — basic frequentist tests
v2/    + Bayesian analysis + logistic regression
v3/    + Power analysis + segmentation + tests
v4/    Dash app — full interactive platform
```

---

## 🚀 Quick Start (v4)

```bash
cd v4
pip install -r requirements.txt
python run.py --dev
```

---

## 📄 License

MIT
