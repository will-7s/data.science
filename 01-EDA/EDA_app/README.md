# Exploratory Data Analysis Web App

An interactive web application for exploratory data analysis (EDA), *"fast"* data cleaning (for visualization and descriptive statistics), and data visualization.  
The app helps users inspect datasets, explore distributions, identify relationships, and generate meaningful insights through interactive charts and filters.

## Features

- Upload and analyze tabular datasets (CSV).
- Perform quick data inspection: shape, data types, missing values, duplicates, and summary statistics.
- Explore univariate and bivariate analysis.
- Visualize distributions with histograms, bar charts, box plots, scatter plots, and correlation heatmaps.
- Interact with charts using filters between columns.
- Reuse insights for reporting and decision-making.

## Why this app?

Exploratory data analysis is the first step in turning raw data into actionable knowledge.  
This application is designed to make that process faster, easier, and more interactive for analysts, students, and decision-makers.
It is a first key step that allows a fast analysis of the dataset and to draw meaninful insights 

## Technology Stack

- **Python**
- **Dash** for the web application interface
- **Plotly** for interactive visualizations
- **NumPy** for numerical operations
- **SciPy** for statistical analysis, if needed
- A cloud platform for deployment

## Project Structure

```text
project-root/
│
├── eda_app.py
├── statistical_tests.py
├── callbacks.py
├── data_loader.py
├── plots.py
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

Make sure you have:
- Python 3.10 or later
- pip
- Git

### Local setup

```bash
git clone https://github.com/will-7s/data.science.git
cd 01-EDA/v1/EDA_app/       #or v2 ... depending on the version
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app locally

```bash
python eda_app.py
```

Then open the app in your browser at:

```text
http://127.0.0.1:8050
```

## How to use

1. Upload your dataset or load one from the available examples.
2. Review the dataset overview: columns, types, missing values, duplicates, and summary metrics.
3. Select the variables you want to explore.
4. Interact with charts using the filters and dropdowns.
5. Interpret the patterns, correlations, and trends shown by the dashboard.

## Example questions the app can answer

- How are the variables distributed?
- What variables show to have a good correlation?
- Which features contain missing values?
- What is the percentage of potential outliers?
- Are there correlations between numerical variables?
- Which categories dominate the dataset?
- Is there a relationship between a variable and the outcome?


## Data and analysis workflow

The app follows a simple workflow:

1. Data inspection
2. Data cleaning
3. Exploratory analysis
4. Visualization
5. Interpretation and presentation

This approach helps users move from raw data to clear business insights quickly.

## Contributing

Contributions are welcome.

If you would like to contribute:
- Fork the repository.
- Create a new branch.
- Make your changes.
- Open a pull request with a clear description.

Please ensure that your code is clean, documented, and tested before submission.

## Roadmap

Planned improvements may include:
- Advanced filtering and search
- Automatic insight generation
- Support for more file formats
- Machine learning integration

## License

This project is licensed under the MIT License.

## Contact

For questions, suggestions, or bug reports, please open an issue or contact the project maintainer.
