import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("./data/train.csv")

train, test = train_test_split(df, test_size=0.25)

print(df.head())
train.to_csv("data/traindata.csv", index=False)
test.to_csv("data/valdata.csv", index=False)