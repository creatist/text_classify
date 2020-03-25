import pandas as pd 

df = pd.read_csv("../data/data_preproced.csv")
print(df["label"].value_counts().to_dict())