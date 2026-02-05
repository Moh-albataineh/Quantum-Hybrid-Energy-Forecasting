import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(filepath, use_time_features=False):
    # 1. Load Data
    df = pd.read_csv(filepath, index_col='Datetime', parse_dates=True)
    df.sort_index(inplace=True)
    
    # 2. Basic Feature (Load)
    # نستخدم Scaler خاص للحمل فقط لنستطيع عكسه لاحقاً
    load_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_load = load_scaler.fit_transform(df[['PJME_MW']])
    
    if not use_time_features:
        # السيناريو القديم (V1.3)
        return scaled_load, load_scaler, 1 # 1 = Input Dim
    
    # 3. Time Features (The Revolution)
    # تحويل الوقت إلى إشارات دائرية (Cyclical Encoding)
    # هذا يجعل الموديل يفهم أن الساعة 23 قريبة من 00
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    # Hour (24 hours)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of Week (7 days)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # تجميع البيانات: العمود الأول هو الحمل (للتوقع)، والباقي مساعدات
    features = np.concatenate([scaled_load, 
                               df[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']].values], axis=1)
    
    return features, load_scaler, 5 # 5 = Input Dim (1 Load + 4 Time)