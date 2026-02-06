import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(filepath, use_time_features=False):
    """
    Loads and processes the energy consumption data.
    
    Args:
        filepath (str): Path to the CSV file.
        use_time_features (bool): If True, adds cyclical time encoding (Hour, Day).
        
    Returns:
        features (np.array): Processed data array.
        load_scaler (MinMaxScaler): Scaler object for the target variable (to inverse transform later).
        input_dim (int): Number of input features (1 or 5).
    """
    # 1. Load Data
    df = pd.read_csv(filepath, index_col='Datetime', parse_dates=True)
    df.sort_index(inplace=True)
    
    # 2. Basic Feature (Load)
    # Initialize a specific scaler for the 'Load' column to facilitate inverse transformation later
    load_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_load = load_scaler.fit_transform(df[['PJME_MW']])
    
    if not use_time_features:
        # Legacy Scenario (V1.3): Return only the load sequence
        return scaled_load, load_scaler, 1 # 1 = Input Dimension
    
    # 3. Time Features (Cyclical Encoding)
    # Converting time indices into cyclical sine/cosine signals.
    # This allows the neural network to understand that Hour 23 is close to Hour 0.
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    # Hour Encoding (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of Week Encoding (7-day cycle)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Feature Concatenation: 
    # Column 0 is the Target (Load), remaining columns are Time Embeddings.
    features = np.concatenate([
        scaled_load, 
        df[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']].values
    ], axis=1)
    
    return features, load_scaler, 5 # 5 = Input Dimension (1 Load + 4 Time Features)