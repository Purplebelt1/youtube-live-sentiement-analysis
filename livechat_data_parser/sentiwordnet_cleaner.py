import pandas as pd
from tqdm import tqdm

sentiwordnet_df = pd.read_csv("livechat_data_parser/sentiwordnet.txt", sep = '\t')

transformed_df = pd.DataFrame(columns=['Word', 'POS', 'PosScore', 'NegScore'])

# Function to split synset terms and create rows for each word
def split_synset_terms(row):
    synset_terms = row['SynsetTerms'].split()
    pos = row['POS']  # Get POS from the original DataFrame
    for word in synset_terms:
        word = word.split('#')[0]
        if word in transformed_df['Word'].values:
            # If word already exists, update scores by averaging
            current_row = transformed_df[transformed_df['Word'] == word]
            new_pos_score = (current_row['PosScore'].values[0] + row['PosScore']) / 2
            new_neg_score = (current_row['NegScore'].values[0] + row['NegScore']) / 2
            transformed_df.loc[transformed_df['Word'] == word, ['PosScore', 'NegScore']] = [new_pos_score, new_neg_score]
        else:
            transformed_df.loc[len(transformed_df)] = [word, pos, row['PosScore'], row['NegScore']]

    pbar.update(1)

# Apply function row-wise to split synset terms and create rows
# Use tqdm to create a progress bar
with tqdm(total=len(sentiwordnet_df)) as pbar:
    sentiwordnet_df.apply(split_synset_terms, axis=1)

transformed_df['SentScore'] = transformed_df['PosScore'] - transformed_df['NegScore']

transformed_df = transformed_df[['Word', 'POS', 'SentScore']]

# Save transformed data to CSV
transformed_df.to_csv("livechat_data_parser/cleaned_sentiwordnet.txt", sep=',', index=False, encoding='utf-8')
