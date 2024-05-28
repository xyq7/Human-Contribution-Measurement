import pandas as pd
import numpy as np
import json
# Path to the CSV file
csv_file_path = './raw_data_new/NewsArticles.csv'
df = pd.read_csv(csv_file_path)

# Ensure there is an 'article_id' column
if 'article_id' not in df.columns:
    raise ValueError("CSV file must contain 'article_id' column")

# Add a column for word counts in the text to filter properly
df['text_word_count'] = df['text'].str.split().str.len()
df['title_word_count'] = df['title'].str.split().str.len()

# Filter to keep texts within 400 to 700 words and titles with more than 3 words
filtered_df = df[(df['text_word_count'] >= 400) & (df['text_word_count'] <= 700) & (df['title_word_count'] > 3)]

# Sample up to 2000 entries, less if not available
n_samples = min(2000, len(filtered_df))
sampled_df = filtered_df.sample(n=n_samples, random_state=1)

# Convert sampled data to the desired dictionary format using 'article_id' as the key
formatted_random_data = {}
for _, row in sampled_df.iterrows():
    # import pdb; pdb.set_trace()
    formatted_random_data[row['article_id']] = {
        'title': row['title'].strip(),
        'abstract': row['text'].strip(),
         }
# Save the formatted data to a JSON file
with open('./data_new/news.json', 'w', encoding='utf-8') as outfile:
    json.dump(formatted_random_data, outfile, indent=4)

# Calculate and print the mean and standard deviation of word counts for text and titles
title_word_counts = [len(item['title'].split()) for item in formatted_random_data.values()]
text_word_counts = [len(item['abstract'].split()) for item in formatted_random_data.values()]

text_mean = np.mean(text_word_counts)
text_std = np.std(text_word_counts)
title_mean = np.mean(title_word_counts)
title_std = np.std(title_word_counts)

print(f"Mean word count for texts: {text_mean} ± {text_std}")
print(f"Mean word count for titles: {title_mean} ± {title_std}")