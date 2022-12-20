import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms


df = pd.read_csv('data/total_labels.csv')
df['id'] = df['id'].apply(str)
labels = dict(zip(df.id, df.boneage))
label_gender = dict(zip(df.id ,df.male))
data = list(pd.read_csv("processed_data/total_train.csv", header=None)[0])

positives = [a for a in data if label_gender[str(a)] == True]
negatives = [a for a in data if label_gender[str(a)] == False]


print(len(positives))
print(len(negatives))

