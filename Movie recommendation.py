import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Get the dataset

columns_name = ["user_id","item_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep ='\t', names = columns_name) #seperator is /t - tab
print(df.head())
print(df.shape) 

movies_title = pd.read_csv("ml-100k/u.item", sep ='\|', header = None , encoding = "ISO-8859-1")
print(movies_title)
