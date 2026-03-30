import numpy as np
from scipy import stats

def normality_test(data):
    """Perform Shapiro-Wilk test for normality"""
    data_clean = data[~np.isnan(data)]
    if len(data_clean) < 3 or len(data_clean) > 5000:
        return None, None
    try:
        statistic, p_value = stats.shapiro(data_clean)
        return statistic, p_value
    except:
        return None, None

def get_normality_interpretation(p_value):
    """Interpret normality test results"""
    if p_value is None:
        return "Insufficient data or Numerical Instability"
    elif p_value > 0.05:
        return f"Normal (p={p_value:.3f})"
    else:
        return f"Non-normal (p={p_value:.3f})"

def perform_appropriate_test(var1, var2, loaded_data, column_types):
    """Perform the most appropriate statistical test"""
    
    if var1 not in loaded_data or var2 not in loaded_data:
        return {'test_name': 'Error', 'results': ['Variables not found']}
    
    data1 = loaded_data[var1]
    data2 = loaded_data[var2]
    
    var1_type = column_types.get(var1, 'categorical')
    var2_type = column_types.get(var2, 'categorical')
    
    if var1_type == 'numeric' and var2_type == 'numeric':
        mask = ~(np.isnan(data1) | np.isnan(data2))
        clean1 = data1[mask]
        clean2 = data2[mask]
        
        if len(clean1) > 1:
            pearson_corr, pearson_p = stats.pearsonr(clean1, clean2)
            spearman_corr, spearman_p = stats.spearmanr(clean1, clean2)
            _, norm_p = normality_test(clean1)
            is_normal = norm_p > 0.05 if norm_p else False
            
            return {
                'test_name': 'Correlation Analysis',
                'results': [
                    f"Pearson: {pearson_corr:.3f} (p={pearson_p:.4f})",
                    f"Spearman: {spearman_corr:.3f} (p={spearman_p:.4f})",
                    f"Use: {'Pearson' if is_normal else 'Spearman'}"
                ]
            }
    
    elif (var1_type == 'numeric' and var2_type == 'categorical') or \
         (var1_type == 'categorical' and var2_type == 'numeric'):
        
        num_var = var1 if var1_type == 'numeric' else var2
        cat_var = var2 if var1_type == 'numeric' else var1
        
        num_data = loaded_data[num_var]
        cat_data = loaded_data[cat_var]
        
        unique_cats = np.unique(cat_data)
        groups = []
        for cat in unique_cats:
            group_data = num_data[cat_data == cat]
            group_data = group_data[~np.isnan(group_data)]
            if len(group_data) > 0:
                groups.append(group_data)
        
        if len(groups) >= 2:
            f_stat, anova_p = stats.f_oneway(*groups)
            h_stat, kruskal_p = stats.kruskal(*groups)
            
            return {
                'test_name': 'Group Comparison',
                'results': [
                    f"ANOVA: F={f_stat:.3f}, p={anova_p:.4f}",
                    f"Kruskal-Wallis: H={h_stat:.3f}, p={kruskal_p:.4f}",
                    f"Use: {'ANOVA' if anova_p > 0.05 else 'Kruskal-Wallis'}"
                ]
            }
    
    else:
        unique1 = np.unique(data1)
        unique2 = np.unique(data2)
        
        contingency = np.zeros((len(unique1), len(unique2)))
        for i, val1 in enumerate(unique1):
            for j, val2 in enumerate(unique2):
                contingency[i, j] = np.sum((data1 == val1) & (data2 == val2))
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        n = np.sum(contingency)
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        return {
            'test_name': 'Chi-square Test',
            'results': [
                f"Chi-square: {chi2:.3f} (p={p_value:.4f})",
                f"Cramer's V: {cramers_v:.3f}",
                f"Association: {'Yes' if p_value < 0.05 else 'No'}"
            ]
        }
    
    return {'test_name': 'Error', 'results': ['Insufficient data']}

def calculate_correlation_matrix(loaded_data, numeric_columns):
    """Calculate pairwise correlation matrix"""
    if len(numeric_columns) < 2:
        return None, None
    
    n_vars = len(numeric_columns)
    corr_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                col1 = loaded_data[numeric_columns[i]]
                col2 = loaded_data[numeric_columns[j]]
                mask = ~(np.isnan(col1) | np.isnan(col2))
                if np.sum(mask) > 1:
                    corr_matrix[i, j] = np.corrcoef(col1[mask], col2[mask])[0, 1]
                else:
                    corr_matrix[i, j] = 0
    
    return corr_matrix, numeric_columns