import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


df = pd.read_csv('data/labels.csv',  names=['image', 'label'], skiprows=1)
df = df.iloc[:, 0]
df = df.values.tolist()






label_0 = []
label_1 = []
label_2 = []
label_3 = []
label_4 = []


print(df)

print(df[0][-8])
label_0 = [a for a in df if a[-10:-9] == '_'] 


print(len(label_0))