import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import StanfordTagger



nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')


def prelim_clean(sentiment_data):

    sentiment_data = sentiment_data[(sentiment_data['sentiment'] != "") &
                                    sentiment_data['sentiment'].notna() &
                                    sentiment_data['content.text_content'].notna()]

    sentiment_data['content.text_content'] = sentiment_data['content.text_content'].str.lower()

    punc = ["?", ".", ",", "!"]

    def count_punctuation(text):
        punctuation_counts = {p: 0 for p in punc}
        for char in text:
            if char in punc:
                punctuation_counts[char] += 1
        
        last_punctuation = None
        for char in reversed(text):
            if char in punc:
                if char == ",":
                    char = "comma"
                last_punctuation = char
                break

        punctuation_counts["comma"] = punctuation_counts.pop(",")
        
        return pd.Series({
            'last_punctuation': last_punctuation,
            **{f'{p}_count': count for p, count in punctuation_counts.items()}
        })

    # Apply the function to each row of the DataFrame using .apply() and store the results in a new DataFrame
    punctuation_counts_df = sentiment_data['content.text_content'].apply(count_punctuation)

    # Combine the original DataFrame with the new DataFrame containing punctuation counts and the last punctuation
    sentiment_data = pd.concat([sentiment_data, punctuation_counts_df], axis=1)

    # Remove non-alphabetic characters using regular expressions

    sentiment_data['content.text_content'] = sentiment_data['content.text_content'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Keep only 'content.text_content' and 'sentiment' columns
    sentiment_data = sentiment_data[['content.text_content', 'sentiment', 'last_punctuation', '?_count', '._count', 'comma_count', '!_count']]

    sentiment_data['content.text_content'] = sentiment_data['content.text_content'].str.strip()

    sentiment_data = sentiment_data[(sentiment_data['content.text_content'] != "") &
                                    sentiment_data['content.text_content'].notna()]

    # Reset the index after filtering
    sentiment_data.reset_index(drop=True, inplace=True)

    return sentiment_data



def add_to_lexicon(df, lexicon_filepath, output_lexicon_filepath):

    def find_words_not_in_lexicon(data, lexicon, col_name):
        words_not_in = []
        words_in = []
        for i in data[col_name]:
            for j in i.split(" "):
                if ~lexicon['Word'].isin([j]).any() and j not in words_not_in and j != "":
                    words_not_in.append(j)
                elif j not in words_not_in and j != "":
                    words_in.append(j)
        return {"not_in_lex": words_not_in, "in_lex": words_in}
    #sentiment_lexicon = pd.read_csv("livechat_data_parser/cleaned_sentiwordnet.txt", sep = ',')
    sentiment_lexicon = pd.read_csv(lexicon_filepath, sep = ',')

    new_words = find_words_not_in_lexicon(df, sentiment_lexicon, "content.text_content")["not_in_lex"]



    hand_picked_pos = pd.read_csv("livechat_data_parser/handpicked_pos.csv", sep = ',')

    new_words_df = pd.DataFrame(new_words, columns=["text"])

    print(hand_picked_pos['Word'])

    new_words = find_words_not_in_lexicon(new_words_df, hand_picked_pos, "text")

    hand_picked_words = new_words["in_lex"]

    print(hand_picked_words)

    new_words = new_words["not_in_lex"]

    # Tokenize the words
    new_words_tokenized = [word_tokenize(word) for word in new_words]


    # Filter out the stop words
    filtered_words = [[word for word in words if word.lower() not in stop_words] for words in new_words_tokenized]


    tagged_words = [nltk.pos_tag(words) for words in filtered_words]


    # For sentiWordNet:
    # N = noun, V = Verb, R = Adverb, A = Adjective

    # For NLTK:
    # JJ, JJR, JJS = Adjective
    # NN, NNS, NNP, NNPS, PRP, PRP$ = Noun
    # RB, RBR, RBS = Adverb
    # VB, VBG, VBD, VBN, VBP, VBZ = Verb
    # All else is NA


    # Define mapping from NLTK tags to SentiWordNet tags
    nltk_to_swn = {
        'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
        'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n', 'PRP': 'n', 'PRP$': 'n',
        'RB': 'r', 'RBR': 'r', 'RBS': 'r',
        'VB': 'v', 'VBG': 'v', 'VBD': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v'
    }

    # Function to convert NLTK tags to SentiWordNet tags
    def convert_tag(tag):
        return nltk_to_swn.get(tag, 'NA')
    

    swn_tags = [(word, convert_tag(tag)) for sublist in tagged_words for (word, tag) in sublist]

    hand_picked_tagged = [tuple(x) for x in hand_picked_pos[hand_picked_pos["Word"].isin(hand_picked_words)].to_numpy()]

    print(hand_picked_tagged)

    swn_tags = swn_tags + hand_picked_tagged

    word_sentiment_scores = {}

    from collections import defaultdict

    word_sentiment_instances = defaultdict(list)

    # Populate word_sentiment_instances
    for word, tag in swn_tags:
        for i, row in df.iterrows():
            if word in row['content.text_content'].split(" "):
                word_sentiment_instances[(word, tag)].append(row["sentiment"])

    word_avg_sentiment = {}
    for word_tag, sentiments in word_sentiment_instances.items():
        avg_sentiment = sum(sentiments) / len(sentiments)
        word_avg_sentiment[word_tag] = avg_sentiment

    for word_tag, avg_sentiment in word_avg_sentiment.items():
        new_row = {'Word': word_tag[0], 'POS': word_tag[1], 'SentScore': avg_sentiment}
        sentiment_lexicon = pd.concat([sentiment_lexicon,pd.DataFrame([new_row])], ignore_index=True)



    #sentiment_lexicon.to_csv("livechat_data_parser/youtube_lexicon.csv", sep=',', index=False, encoding='utf-8')
    sentiment_lexicon.to_csv(output_lexicon_filepath, sep=',', index=False, encoding='utf-8')

#C:\Users\William\youtube-livechat-scraper-main\livechat_data_parser\training_set_creator.py

directory = 'yt_live_comments/'

lexicon_dir = "livechat_data_parser/cleaned_sentiwordnet.txt"
output_lexicon_dir = "livechat_data_parser/youtube_lexicon.csv"

all_training = pd.DataFrame(columns=['content.text_content', 'sentiment', 'last_punctuation', '?_count', '._count', 'comma_count', '!_count'])
# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Construct the full file path
        filepath = os.path.join(directory, filename)

        df = pd.read_csv(filepath, sep = ',')

        new_clean_df = prelim_clean(df)

        all_training = pd.concat([all_training, new_clean_df], axis = 0, ignore_index=True)

all_training.reset_index(drop=True, inplace=True)
all_training.to_csv("yt_live_comments/prelim_clean/training_prelim_clean.csv", sep=',', index=False, encoding='utf-8')

add_to_lexicon(all_training, lexicon_dir, output_lexicon_dir)

