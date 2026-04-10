# 🔍 Exploratory Data Analysis Web App

An interactive web application for exploratory data analysis (EDA), *"fast"* data cleaning (for visualization and descriptive statistics), and data visualization.  
The app helps users inspect datasets, explore distributions, identify relationships, and generate meaningful insights through interactive charts and filters.

🚀 **Try it live:** [huggingface.co/spaces/will-7s/eda_app](https://huggingface.co/spaces/will-7s/eda_app)

---

## ✨ Features

- 📂 Upload and analyze datasets of several natures
- 🔎 Perform quick data inspection: shape, data types, missing values, duplicates, and summary statistics
- 📊 Explore univariate and bivariate analysis
- 📈 Visualize distributions with histograms, bar charts, box plots, scatter plots, and correlation heatmaps
- 🔗 Interact with charts using filters between columns
- 💡 Reuse insights for reporting and decision-making

---

## 🧭 Workflow

```
 Upload dataset  →  Inspect  →  Clean  →  Explore  →  Visualize  →  Interpret
```

The app follows a structured pipeline that moves from raw data to clear business insights quickly — without writing a single line of code.

| Step | What happens |
|------|-------------|
| **1. Data inspection** | Shape, types, missing values, duplicates, summary stats |
| **2. Data cleaning** | Handle nulls, remove duplicates, fix types |
| **3. Univariate analysis** | Distributions, statistics, normality per variable |
| **4. Bivariate analysis** | Correlations, relationships, associations |
| **5. Visualization** | Interactive charts with cross-column filters |
| **6. Interpretation** | Export insights for reporting and decision-making |

---

## ❓ Example Questions the App Can Answer

- How are the variables distributed?
- Which features contain missing values?
- What is the percentage of potential outliers?
- Are there correlations between numeric variables?
- Which categories dominate the dataset?
- Is there an association between two categorical variables?
- Is there a relationship between a variable and the outcome?

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **Web framework** | Dash |
| **Visualizations** | Plotly |
| **Numerical operations** | NumPy |
| **Statistical analysis** | SciPy |
| **Deployment** | Hugging Face Spaces |

---

## 📁 Project Structure

```text
project-root/
│
├── eda_app.py        ← App entry point
├── stats.py          ← Statistical computations
├── callbacks.py      ← Dash interactivity logic
├── loader.py         ← Data loading utilities
├── parsers.py        ← File parsing (CSV, Excel, ...)
├── charts.py         ← Chart generation
├── store.py          ← State management
├── ui.py             ← Layout and UI components
├── utils.py          ← Helper functions
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or later
- pip
- Git

### Local setup

```bash
git clone https://github.com/will-7s/data.science.git
cd 01-EDA/v3/EDA_app/       # or v4 ... depending on the last version
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run locally

```bash
python eda_app.py
```

Then open your browser at:

```
http://127.0.0.1:8050
```

---

## 🗺️ How to Use

1. **Upload** your dataset or load one from the available examples
2. **Review** the dataset overview: columns, types, missing values, duplicates, and summary metrics
3. **Select** the variables you want to explore
4. **Interact** with charts using the filters and dropdowns
5. **Interpret** the patterns, correlations, and trends shown by the dashboard

---

## 🔮 Roadmap

Planned improvements may include:

- [ ] Advanced filtering and search
- [ ] Automatic insight generation
- [ ] Support for more file formats
- [ ] Machine learning integration

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Open a pull request with a clear description

Please ensure your code is clean, documented, and tested before submission.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

For questions, suggestions, or bug reports, please open an issue or contact the project maintainer.
