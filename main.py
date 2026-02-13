import pandas as pd

from src.config import *
from src.load_data import load_dataset
from src.preprocess import preprocess_dataset
from src.feature_selection import feature_selection
from src.train_model import train_model

def main():
    df = load_dataset()
    new_df = preprocess_dataset(df)
    X_train,X_test,y_train,y_test = feature_selection(new_df)
    train_model(X_train,X_test,y_train,y_test)    
if __name__ == '__main__':
    main()