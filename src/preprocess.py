import pandas as pd
import numpy as np
from src.config import *

def preprocess_dataset(df:pd.DataFrame): 
    df['volume_m'] = (df['length_mm']*df['width_mm']*df['height_mm'])/1000000000
    df['cargo_volume'] = pd.to_numeric(df['cargo_volume_l'], errors = 'coerce')
    df['cargo_volume'].isnull().sum()
    df.drop(columns=['source_url','height_mm','width_mm','length_mm','cargo_volume_l'], inplace=True)
    for col in ['number_of_cells', 'torque_nm', 'towing_capacity_kg', 'cargo_volume']:
        df[col] = df[col].fillna(df[col].median())
    
    Q1 = df['number_of_cells'].quantile(0.25)
    Q3 = df['number_of_cells'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['number_of_cells'] = np.clip(df['number_of_cells'], lower_bound, upper_bound)
    df.dropna(inplace=True)
    numerical_columns = [column for column in df.columns if df[column].dtype != 'O']
    df = df[numerical_columns]
    corr_matrix = df.corr()
    threshold=0.5
    target_corr = corr_matrix['range_km'].abs()
    features = target_corr[target_corr >= threshold].index.to_list()
    df = df[features]
    print("preprocess_dataset ok!")
    return df

    
    
