import pandas as pd



df = pd.read_csv("livechat_data_parser/youtube_lexicon.csv", sep = ',')

df = df.iloc[:, :2]

df = df[df.index >= 147308]



df.to_csv("livechat_data_parser/quicky.csv", sep=',', index=False, encoding='utf-8')
