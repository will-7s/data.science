# EDA Dashboard Application - v1

This is a professional, multi-component web application built with Dash and Python that allows users to upload data files and perform automated Exploratory Data Analysis (EDA). The application handles both numeric and categorical data, generates visualizations, and runs appropriate statistical tests.

Below is a breakdown of each file's purpose in the overall system.

---

## 📁 `eda_app.py`
**Role:** Application Entry Point & Layout Definition

This is the main file that launches the dashboard. It defines the user interface layout using Bootstrap components, including:
- The file upload section
- Tabs for **Univariate** (single variable) and **Bivariate** (two variables) analysis
- Dropdown menus, graph placeholders, and statistics panels

It initializes the Dash application and connects all visual components to the backend logic defined in other files.

---

## 📁 `callbacks.py`
**Role:** Interactive Logic & Event Handling

This file acts as the "brain" of the application. It contains callback functions that define what happens when a user interacts with the interface (e.g., uploading a file or selecting a variable).

Key responsibilities:
- Triggering analysis when a file is uploaded
- Updating dropdown options based on the loaded data
- Generating plots and statistics when a user selects a variable or changes a chart type
- Coordinating between the frontend (user interface) and backend (data processing & statistics)

---

## 📁 `data_loader.py`
**Role:** Data Loading & Preprocessing

This module handles everything related to reading and preparing the user's data file.

Capabilities:
- Supports **CSV** and **Excel** files
- Automatically detects whether a column is **numeric** (e.g., age, price) or **categorical** (e.g., gender, country)
- Removes duplicate rows
- Converts data into a format suitable for analysis (NumPy arrays)
- Stores the loaded data globally for other modules to access

---

## 📁 `statistical_tests.py`
**Role:** Statistical Analysis Engine

This file provides all the mathematical and statistical logic used in the application.

Functions include:
- **Normality testing** (Shapiro-Wilk test) to check if numeric data follows a normal distribution
- **Correlation analysis** (Pearson & Spearman) for two numeric variables
- **Group comparison** (ANOVA & Kruskal-Wallis) for numeric vs. categorical variables
- **Chi-square test** for two categorical variables, including Cramer’s V for effect size
- **Correlation matrix** calculation for multiple numeric variables

Each test includes automatic recommendations on which result to trust based on data characteristics.

---

## 📁 `plots.py`
**Role:** Visualization Generation

This file is responsible for creating all the charts and graphs displayed in the dashboard.

Visualization types include:
- **Histograms**, **box plots**, and **bar charts** for univariate analysis
- **Scatter plots** for two numeric variables
- **Box plots** for numeric vs. categorical data
- **Heatmaps** for categorical-categorical relationships
- **Correlation matrix heatmaps**

All plots use Plotly, making them interactive (zoom, hover, pan).

---

## 📁 `requirements.txt`
**Role:** Dependency Management

This file lists every Python library required to run the application, along with their specific versions.

Key dependencies include:
- `dash` – web framework
- `plotly` – interactive graphing
- `pandas`, `numpy` – data handling
- `scipy`, `statsmodels` – statistical tests
- `gunicorn` – for production deployment

To install all dependencies at once:
```bash
pip install -r requirements.txt