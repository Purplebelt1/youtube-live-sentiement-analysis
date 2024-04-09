import pandas as pd
from tqdm import tqdm

yt_lexicon = pd.read_csv("livechat_data_parser/youtube_lexicon.csv", sep = ',')
sentiment_data = pd.read_csv("yt_live_comments/prelim_clean/training_prelim_clean.csv", sep = ',')

#print(sentiment_data.columns)

sentiment_data_pos = pd.DataFrame(columns=['message', 'sentiment', 'avg_noun_sentiment', 'avg_verb_sentiment', 'avg_adverb_sentiment', 'avg_adjective_sentiment', 'avg_na_sent', 'avg_tot_sent', 'sum_noun_sentiment', 'sum_verb_sentiment', 'sum_adverb_sentiment', 'sum_adjective_sentiment', 'last_punctuation', 'comma_count', '._count', ',_count', '!_count'])

def add_pos(row):
    noun_sent, noun_count, verb_sent, na_sent, na_count, verb_count, adverb_sent, adverb_count, adj_sent, adj_count = [0] * 10

    for row_word in row['content.text_content'].split(" "):
        word_info = yt_lexicon[(yt_lexicon['Word'] == row_word)]
        if not word_info.empty:
            pos = word_info['POS'].iloc[0]
            score = word_info['SentScore'].iloc[0]
            if pos == 'n':
                noun_sent += score
                noun_count += 1
            elif pos == 'v':
                verb_sent += score
                verb_count += 1
            elif pos == 'r':
                adverb_sent += score
                adverb_count += 1
            elif pos == 'a':
                adj_sent += score
                adj_count += 1
            else:
                na_sent += score
                na_count += 1
    pbar.update(1)

    def div_not_zero(x,y):
        if y == 0:
            return 0
        return x/y

    # Calculate average sentiment for each word class
    tot_sent = noun_sent + verb_sent + adverb_sent + adj_sent + na_sent
    tot_count = noun_count + verb_count + adverb_count + adj_count + na_count
    noun_avg_sent = div_not_zero(noun_sent,noun_count)
    verb_avg_sent = div_not_zero(verb_sent,verb_count)
    adverb_avg_sent = div_not_zero(adverb_sent,adverb_count)
    adj_avg_sent = div_not_zero(adj_sent,adj_count)
    na_avg_sent = div_not_zero(na_sent,na_count)
    tot_avg_sent = div_not_zero(tot_sent, tot_count)

    # Access punctuation count columns from the row object
    sentiment_data_pos.loc[len(sentiment_data_pos)] = [
        row['content.text_content'], 
        row['sentiment'],
        noun_avg_sent, 
        verb_avg_sent, 
        adverb_avg_sent, 
        adj_avg_sent,
        na_avg_sent,
        tot_avg_sent,
        noun_sent,
        verb_count,
        adverb_count,
        adj_count,
        row['last_punctuation'],
        row['?_count'],
        row['._count'], 
        row['comma_count'], 
        row['!_count']
    ]

with tqdm(total=len(sentiment_data)) as pbar:
    sentiment_data.apply(add_pos, axis=1)

sentiment_data_pos.to_csv("yt_live_comments/final_clean/training_clean.csv", sep=',', index=False, encoding='utf-8')