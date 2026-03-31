import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def create_histogram(data, variable):
    """Create histogram for numeric variable"""
    data_clean = data[~np.isnan(data)]
    fig = px.histogram(x=data_clean, color_discrete_sequence=['#3498db'])
    fig.update_layout(template='plotly_white', height=500, margin=dict(l=40, r=40, t=20, b=40))
    return fig

def create_boxplot(data, variable):
    """Create box plot for numeric variable"""
    data_clean = data[~np.isnan(data)]
    fig = px.box(y=data_clean, color_discrete_sequence=['#e74c3c'])
    fig.update_layout(template='plotly_white', height=500, margin=dict(l=40, r=40, t=20, b=40))
    return fig

def create_barchart_numeric(data, variable):
    """Create bar chart for binned numeric data"""
    data_clean = data[~np.isnan(data)]
    hist, bins = np.histogram(data_clean, bins=20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    fig = px.bar(x=bin_centers, y=hist, color_discrete_sequence=['#2ecc71'])
    fig.update_layout(template='plotly_white', height=500, margin=dict(l=40, r=40, t=20, b=40))
    return fig

def create_barchart_categorical(data, variable):
    """Create bar chart for categorical variable"""
    unique_vals, counts = np.unique(data, return_counts=True)
    fig = px.bar(x=unique_vals, y=counts, labels={'x': variable, 'y': 'Frequency'},
                 color_discrete_sequence=['#3498db'])
    fig.update_layout(template='plotly_white', height=500, margin=dict(l=40, r=40, t=20, b=40))
    return fig

def create_scatter_plot(x_data, y_data, x_label, y_label):
    """Create scatter plot for two numeric variables"""
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    fig = px.scatter(x=x_data[mask], y=y_data[mask], labels={'x': x_label, 'y': y_label},
                     opacity=0.6, color_discrete_sequence=['#3498db'])
    fig.update_layout(template='plotly_white', height=400, margin=dict(l=40, r=40, t=20, b=40))
    return fig

def create_boxplot_bivariate(num_data, cat_data, num_label, cat_label):
    """Create box plot for numeric vs categorical"""
    unique_cats = np.unique(cat_data)
    fig = go.Figure()
    for cat in unique_cats:
        values = num_data[cat_data == cat]
        values_clean = values[~np.isnan(values)]
        if len(values_clean) > 0:
            fig.add_trace(go.Box(y=values_clean, name=str(cat)))
    fig.update_layout(xaxis_title=cat_label, yaxis_title=num_label,
                      template='plotly_white', height=400, margin=dict(l=40, r=40, t=20, b=40))
    return fig

def create_heatmap(data1, data2, var1, var2):
    """Create heatmap for two categorical variables"""
    unique1 = np.unique(data1)
    unique2 = np.unique(data2)
    
    contingency = np.zeros((len(unique1), len(unique2)))
    for i, val1 in enumerate(unique1):
        for j, val2 in enumerate(unique2):
            contingency[i, j] = np.sum((data1 == val1) & (data2 == val2))
    
    row_sums = contingency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    contingency_pct = contingency / row_sums * 100
    
    fig = px.imshow(contingency_pct, x=unique2.astype(str), y=unique1.astype(str),
                    labels=dict(x=var2, y=var1, color="%"),
                    color_continuous_scale="Blues", aspect="auto", text_auto=True)
    fig.update_layout(template='plotly_white', height=400, margin=dict(l=40, r=40, t=20, b=40))
    return fig

def create_correlation_matrix(corr_matrix, columns):
    """Create correlation matrix heatmap"""
    fig = px.imshow(corr_matrix, x=columns, y=columns, text_auto=True,
                    aspect="auto", color_continuous_scale="RdBu_r",
                    labels=dict(color="Correlation"), zmin=-1, zmax=1)
    fig.update_layout(template='plotly_white', height=500, margin=dict(l=40, r=40, t=20, b=40))
    return fig

def create_empty_figure(message="No data"):
    """Create empty figure with message"""
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=400)
    return fig