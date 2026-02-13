import pandas as pd

from src.config import *

def load_dataset()-> pd.DataFrame:
    data_set = pd.read_csv(DATASET)
    print("load_dataset ok!")
    return data_set
