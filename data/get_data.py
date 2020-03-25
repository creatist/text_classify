import pandas as pd 

df = pd.read_csv("check_result.csv")
df = df[["ctype1", "status", "text"]]


ctype1s = set(df["ctype1"])
print("ctype1", ctype1s)

df["label"] = df.apply(lambda x:"IGNORE" if x["status"]=="IGNORE" and pd.isna(x["ctype1"]) else x["ctype1"], axis=1 )
# df.apply(lambda x:x, axis=0 )

df = df[["ctype1", "status", "label","text"]]

df.to_csv("data.csv", index=False)