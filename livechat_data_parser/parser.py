# Load JSON data

from PIL import Image as PILImage
from IPython.display import display, Image
import os
import re
from wordcloud import WordCloud
from livechat_message import LiveChatMessage
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from scipy.special import softmax
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

from reader import load_dataframe

HF_ROBERTA_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
hf_tokenizer = AutoTokenizer.from_pretrained(HF_ROBERTA_MODEL)
hf_model = AutoModelForSequenceClassification.from_pretrained(HF_ROBERTA_MODEL)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')


plt.style.use('ggplot')
#data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

#data_dir = str(data_dir)

data_dir = "yt_live_comments/prelim_clean/training_prelim_clean.csv"

print(data_dir)

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
Lda = gensim.models.ldamodel.LdaModel

def open_data_file(input_file):
    if input_file[-4:] == "json":

        df = load_dataframe(input_file)

        sentiment = [None for i in range(len(df))]

        df["sentiment"] = sentiment

    elif input_file[-3:] == "csv":
        df = pd.read_csv(input_file)
    
    return df

data = open_data_file(data_dir)

#with open(data_dir + "\\json_test.json", 'r', encoding='utf-8') as file:
#    messages = json.load(file)

#messages = messages[10000:10300]
# Extract text content from each message

sia = SentimentIntensityAnalyzer()

# Cleaning code for Topic Modeling from https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
def clean_for_topic_modeling(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


message_info = []


for message in tqdm(data["content.text_content"]):
    #print(message)
    #try:
    #    text = message["content"]["message"]
    #except:
    #    continue
    text = message
    # text_content_tokenized = nltk.word_tokenize(text_content)
    # text_content_tagged = nltk.pos_tag(text_content_tokenized)
    # text_content_chunked = nltk.chunk.ne_chunk(text_content_tagged)

    vader_pol_score = sia.polarity_scores(text)

    hf_encoded_content = hf_tokenizer(text, return_tensors='pt')
    hf_model_output = hf_model(**hf_encoded_content)
    hf_model_score = softmax(hf_model_output[0][0].detach().numpy())

    cleaned_tm = clean_for_topic_modeling(text)

    message_info.append(LiveChatMessage(text, vader_pol_score, hf_model_score, cleaned_tm))


cleaner = re.compile('[,\.!?]')


for info in message_info:
    cleaned_message = re.sub(r'\$\w+', '', info.text_content)
    info.cleaned_message = cleaner.sub('', cleaned_message).lower()

# Join the different processed titles together.
long_string = ','.join(list(message.cleaned_message for message in message_info))

cleaned_list = [message.cleaned_tm_list.split() for message in message_info]

tm_dictionary = corpora.Dictionary(cleaned_list)
tm_term_matrix = [tm_dictionary.doc2bow(doc) for doc in cleaned_list]
#ldamodel = Lda(tm_term_matrix, num_topics=10, id2word=tm_dictionary, passes=100)

data_message = pd.DataFrame(columns = ["text", "vader", "roberta"])

for i, live_message in enumerate(message_info):
    data_message.loc[len(data_message)] = [live_message.text_content, live_message.vader_output, live_message.roberta_output]

data_message['ID'] = range(1, len(data_message) + 1)
data_message.to_csv("yt_live_comments/comparison_models/vader_roberta.csv", sep=',', index=False, encoding='utf-8')
