from dash import html, Input, Output
from dash import dependencies, dcc, State
import plotly.graph_objects as go
import numpy as np
from data_loader import loaded_data, numeric_columns, categorical_columns, column_types
from statistical_tests import normality_test, get_normality_interpretation
from statistical_tests import perform_appropriate_test, calculate_correlation_matrix
from plots import *

def register_callbacks(app):
    
    @app.callback(
        [Output('upload-status', 'children'),
         Output('main-content', 'style'),
         Output('univariate-variable', 'options'),
         Output('univariate-variable', 'value'),
         Output('bivariate-var1', 'options'),
         Output('bivariate-var1', 'value'),
         Output('bivariate-var2', 'options'),
         Output('bivariate-var2', 'value')],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')]
    )
    def handle_file_upload(contents, filename):
        from data_loader import load_data_from_file, data_columns, numeric_columns, categorical_columns
        
        if contents is None:
            return "No file uploaded", {'display': 'none'}, [], None, [], None, [], None
        
        success, message = load_data_from_file(contents, filename)
        
        if success:
            options = [{'label': col, 'value': col} for col in data_columns]
            default_var1 = numeric_columns[0] if numeric_columns else data_columns[0]
            default_var2 = numeric_columns[1] if len(numeric_columns) > 1 \
                else (categorical_columns[0] if categorical_columns \
                else data_columns[1] if len(data_columns) > 1 else data_columns[0])
            
            return message, {'display': 'block'}, options, default_var1, options, \
                default_var1, options, default_var2
        else:
            return message, {'display': 'none'}, [], None, [], None, [], None

    @app.callback(
        [Output('univariate-plot', 'figure'),
         Output('univariate-stats', 'children')],
        [Input('univariate-variable', 'value'),
         Input('plot-type', 'value')]
    )
    def update_univariate(variable, plot_type):
        from data_loader import loaded_data, numeric_columns
        
        if variable is None or loaded_data is None or variable not in loaded_data:
            return create_empty_figure(), html.Div("No data loaded")
        
        data = loaded_data[variable]
        
        if variable in numeric_columns:
            if plot_type == 'histogram':
                fig = create_histogram(data, variable)
            elif plot_type == 'box':
                fig = create_boxplot(data, variable)
            else:
                fig = create_barchart_numeric(data, variable)
            
            data_clean = data[~np.isnan(data)]
            stat, p_val = normality_test(data_clean)
            normality_text = get_normality_interpretation(p_val)

               
            q1, q3 = np.quantile(data_clean, [0.25, 0.75])
            iqr = q3 - q1
            coef = 1.5
            lower_iqr = q1 - coef * iqr
            upper_iqr = q3 + coef * iqr
            outliers_mask = (data_clean < lower_iqr) | (data_clean > upper_iqr)
            outliers_percentage = (np.sum(outliers_mask) / data_clean.size) * 100
            
            stats_html = html.Div([
                html.H4(f"Statistics - {variable}"),
                html.Div([
                    html.Div([html.Span("Mean: ", className="stat-label"), 
                              html.Span(f"{np.mean(data_clean):.2f}", className="stat-value")]),
                    html.Div([html.Span("Median: ", className="stat-label"), 
                              html.Span(f"{np.median(data_clean):.2f}", className="stat-value")]),
                    html.Div([html.Span("Std: ", className="stat-label"), 
                              html.Span(f"{np.std(data_clean):.2f}", className="stat-value")]),
                    html.Div([html.Span("Min/Max: ", className="stat-label"), 
                              html.Span(f"{np.min(data_clean):.2f} / {np.max(data_clean):.2f}", 
                                        className="stat-value")]),
                    html.Div([html.Span("Potential Outliers: ", className="stat-label"), 
                              html.Span(f"{(outliers_percentage):.2f} %", 
                                        className="stat-value")]),
                ]),
                html.Hr(),
                html.Div([html.Span("Normality: ", className="stat-label"), 
                          html.Span(normality_text, className="stat-value")])
            ])
        else:
            if plot_type == 'histogram' or plot_type == 'bar':
                fig = create_barchart_categorical(data, variable)
            else:
                fig = create_empty_figure("Box plot not suitable for categorical data")
            
            unique_vals, counts = np.unique(data, return_counts=True)
            most_frequent_idx = np.argmax(counts)
            
            stats_html = html.Div([
                html.H4(f"Statistics - {variable}"),
                html.Div([
                    html.Div([html.Span("Categories: ", className="stat-label"), 
                              html.Span(f"{len(unique_vals)}", className="stat-value")]),
                    html.Div([html.Span("Most frequent: ", className="stat-label"), 
                             html.Span(f"{unique_vals[most_frequent_idx]} \
                                        ({counts[most_frequent_idx]})", className="stat-value")])
                ])
            ])
        
        return fig, stats_html

    @app.callback(
        [Output('bivariate-plot', 'figure'),
         Output('bivariate-stats', 'children'),
         Output('bivariate-tests', 'children'),
         Output('normality-tests', 'children'),
         Output('pairwise-correlation', 'figure'),
         Output('correlation-insights', 'children')],
        [Input('bivariate-var1', 'value'),
         Input('bivariate-var2', 'value')]
    )
    def update_bivariate(var1, var2):
        from data_loader import loaded_data, numeric_columns, column_types
        
        if var1 is None or var2 is None or loaded_data is None:
            empty_fig = create_empty_figure()
            return empty_fig, html.Div(), html.Div(), html.Div(), empty_fig, html.Div()
        
        test_results = perform_appropriate_test(var1, var2, loaded_data, column_types)
        
        tests_html = html.Div([
            html.H4("Statistical Tests"),
            html.Div([html.Span(f"{test_results['test_name']}:", className="stat-label"),
                     html.Div([html.Span(r, className="stat-value") for r in test_results['results']])])
        ])
        
        normality_html = html.Div([
            html.H4("Normality Tests"),
            html.Div([html.Div([html.Span(f"{col}: ", className="stat-label"),
                               html.Span(get_normality_interpretation(normality_test(loaded_data[col])[1]), 
                                         className="stat-value")])
                     for col in numeric_columns])
        ])
        
        var1_type = column_types.get(var1, 'categorical')
        var2_type = column_types.get(var2, 'categorical')
        
        data1 = loaded_data[var1]
        data2 = loaded_data[var2]
        
        if var1_type == 'numeric' and var2_type == 'numeric':
            fig = create_scatter_plot(data1, data2, var1, var2)
            mask = ~(np.isnan(data1) | np.isnan(data2))
            corr_val = np.corrcoef(data1[mask], data2[mask])[0, 1]
            stats_html = html.Div([
                html.H4("Correlation"),
                html.Div([html.Span(f"r = {corr_val:.3f}", className="stat-value")])
            ])
        elif (var1_type == 'numeric' and var2_type == 'categorical') or (var1_type == 'categorical' and \
                                                                         var2_type == 'numeric'):
            num_var = var1 if var1_type == 'numeric' else var2
            cat_var = var2 if var1_type == 'numeric' else var1
            num_data = loaded_data[num_var]
            cat_data = loaded_data[cat_var]
            fig = create_boxplot_bivariate(num_data, cat_data, num_var, cat_var)
            
            unique_cats = np.unique(cat_data)
            stats_html = html.Div([
                html.H4("Statistics by Category"),
                html.Div([html.Div([html.Span(f"{cat}: ", className="stat-label"),
                                   html.Span(f"μ={np.mean(num_data[cat_data == cat]\
                                                          [~np.isnan(num_data[cat_data == cat])]):.0f}",
                                                            className="stat-value")])
                         for cat in unique_cats])
            ])
        else:
            fig = create_heatmap(data1, data2, var1, var2)
            stats_html = html.Div([html.H4("Association"), html.Div("See Chi-square test results")])
        
        corr_matrix, corr_cols = calculate_correlation_matrix(loaded_data, numeric_columns)
        
        if corr_matrix is not None and len(corr_cols) >= 2:
            corr_fig = create_correlation_matrix(corr_matrix, corr_cols)
            n = len(corr_cols)
            correlations = [(corr_cols[i], corr_cols[j], corr_matrix[i, j]) 
                           for i in range(n) for j in range(i+1, n)]
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            insights_html = html.Div([
                html.H4("Top Correlations"),
                html.Div([html.Div(f"{c[0]} & {c[1]}: {c[2]:.3f}") for c in correlations[:5]])
            ])
        else:
            corr_fig = create_empty_figure("Need 2+ numeric variables")
            insights_html = html.Div("No numeric variables")
        
        return fig, stats_html, tests_html, normality_html, corr_fig, insights_html