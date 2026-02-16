from src.config import *

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def feature_selection(df:pd.DataFrame):
    X = df.drop('range_km',axis=1)# Independent features
    y = df['range_km']# Dependent feature
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=45)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("feature selection ok!")
    scaler_file = 'model/scaler.pkl'
    with open (scaler_file, 'wb') as file:
        pickle.dump(scaler, file)
    print(f'Sclaer saved as {scaler_file}')
    return X_train,X_test,y_train,y_test