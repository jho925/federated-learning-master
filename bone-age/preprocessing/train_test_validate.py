import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("old_data/boneage-training-dataset.csv") 


x = df['id']
y= df['boneage']


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.2)


df_train = pd.DataFrame(list(zip(train_x, train_y)), 
               columns =['Id', 'Age']) 



df_test = pd.DataFrame(list(zip(test_x, test_y)), 
               columns =['Id', 'Age']) 

df_train.to_csv('processed_data/train_boneage.csv',index=False)

df_test.to_csv('processed_data/val_boneage.csv',index=False)

