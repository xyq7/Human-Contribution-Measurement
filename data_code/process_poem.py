import pandas as pd
import json
import re 
import numpy as np

csv_file_path = './raw_data_new/PoetryFoundationData.csv'


json_file_path = './data_new/poem_new.json'
def clean_text(text):
    text = re.sub(r'\r\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


df = pd.read_csv(csv_file_path)
df['Poem'] = df['Poem'].apply(clean_text)
df = df[(df['Poem'].str.split().str.len() <= 500) & (df['Poem'].str.split().str.len()>=100)]


n_samples = min(2000, len(df))

sampled_df = df.sample(n=n_samples, random_state=1)

total_abstract_length = 0
total_title_length = 0

json_data = {}

for idx, row in sampled_df.iterrows():
    json_data[idx] = {
        'title':clean_text(row['Title']),
        'abstract': clean_text(row['Poem']),
    }
    total_abstract_length += len(row['Poem'].split())
    total_title_length += len(row['Title'].split())

with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)
# Calculate and print the mean and standard deviation of word counts for text and titles
# text_word_counts = [item['word_count'] for item in formatted_random_data.values()]
title_word_counts = [len(item['title'].split()) for item in json_data.values()]
text_word_counts = [len(item['abstract'].split()) for item in json_data.values()]

text_mean = np.mean(text_word_counts)
text_std = np.std(text_word_counts)
title_mean = np.mean(title_word_counts)
title_std = np.std(title_word_counts)
print(f"Mean word count for texts: {text_mean} ± {text_std}")
print(f"Mean word count for titles: {title_mean} ± {title_std}")