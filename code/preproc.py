import jieba
import re 
import pandas as pd 
from string import punctuation

def split_words(df_):
    df = df_.copy()
    df["text_splited"] = df["text_cleaned"].map(lambda x:" ".join(jieba.cut(x)))
    return df

def clean_bankname(text):
    text = re.sub("[【\[]\D+[】\]]", "", text)
    return text


def clean_punctuation(text):
    punc = punctuation + '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。&～、|\s:：'
    text = re.sub(r"[{}]+".format(punc)," ",text)
    return text

def clean(df_):
    df = df_.copy()
    df["text_cleaned"] = df["text"].map(lambda x: clean_bankname(x))
    df["text_cleaned"] = df["text_cleaned"].map(lambda x: clean_punctuation(x))
    return df 


if __name__ == "__main__":

    df = pd.read_csv("../data/data.csv")
    df.dropna(subset=['text'],inplace=True)
    df.drop_duplicates(subset=['text'],inplace=True)
    df = clean(df)
    df = split_words(df)
    df.to_csv("../data/data_preproced.csv", index=False)

    df = pd.read_csv("../data/data_preproced.csv")
    print("columns", df.columns.tolist())
    print("len df", len(df))
    with open("../data/text.txt" , "w+") as f:
        for i in range(len(df)):
            text = df.loc[i]["text_splited"]
            f.write(text+"\n")