import numpy as np
import pandas as pd
import base64
import io

loaded_data = None
data_columns = []
numeric_columns = []
categorical_columns = []
column_types = {}

def load_data_from_file(contents, filename):
    """Load data from uploaded file (CSV or Excel)"""
    global loaded_data, data_columns, numeric_columns, categorical_columns, column_types
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename.lower():
            data = np.genfromtxt(io.StringIO(decoded.decode('utf-8')), 
                                 delimiter=',', dtype=None, encoding='utf-8', names=True)
            data = np.unique(data, axis=0) 

            loaded_data = {}
            numeric_columns.clear()
            categorical_columns.clear()
            
            for name in data.dtype.names:
                column_data = data[name]
                try:
                    loaded_data[name] = column_data.astype(np.float64)
                    numeric_columns.append(name)
                except (ValueError, TypeError):
                    loaded_data[name] = column_data.astype(str)
                    categorical_columns.append(name)
            
            data_columns = list(loaded_data.keys())
            
        elif 'xls' in filename.lower():
            df_temp = pd.read_excel(io.BytesIO(decoded))
            df_temp = df_temp.drop_duplicates()
            loaded_data = {}
            numeric_columns.clear()
            categorical_columns.clear()
            
            for col in df_temp.columns:
                if df_temp[col].dtype in ['int64', 'float64']:
                    loaded_data[col] = df_temp[col].values.astype(np.float64)
                    numeric_columns.append(col)
                else:
                    loaded_data[col] = df_temp[col].values.astype(str)
                    categorical_columns.append(col)
            data_columns = list(loaded_data.keys())
        
        for col in data_columns:
            column_types[col] = 'numeric' if col in numeric_columns else 'categorical'
        
        return True, f"Loaded {filename}: {len(data_columns)} columns, \
              {len(loaded_data[data_columns[0]])} rows"
    
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_column_data(col_name):
    """Get column data as numpy array"""
    global loaded_data
    if loaded_data and col_name in loaded_data:
        return loaded_data[col_name]
    return np.array([])

def get_unique_values(col_name):
    """Get unique values from a column"""
    data = get_column_data(col_name)
    return np.unique(data) if len(data) > 0 else np.array([])