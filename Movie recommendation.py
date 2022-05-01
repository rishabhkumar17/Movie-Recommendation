import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#get the dataset
df = pd.read_csv("ml-100k/u.data", sep ='\t') #seperator is /t - tab
print(df.head())
print(df.shape) 