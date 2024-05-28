import os
import json
import random
# Path to the directory containing JSON files
directory_path = './raw_data_new/2018'

# Prepare the output dictionary to store selected patents
filtered_data = {}
import numpy as np
import os
import json
import random

def format_title(title):
    # Words to exclude from capitalization
    exclusions = {'and', 'or', 'the', 'a', 'an', 'of', 'in', 'to', 'with', 'is', 'for', 'on', 'by'}
    # Split the title into words and capitalize selectively
    title_words = title.split()
    formatted_title = [word.title() if word.lower() not in exclusions else word.lower() for word in title_words]
    # Ensure the first word is always capitalized
    if formatted_title:
        formatted_title[0] = formatted_title[0].title()
    return ' '.join(formatted_title)

# Traverse the directory and process each JSON file
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)
        
        # Read the content of the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # Retrieve necessary fields from the JSON data
            application_number = data.get('application_number', '')
            title = format_title(data.get('title', ''))
            abstract = data.get('abstract', '')
            
            # Check the length of the abstract by word count
            # import pdb; pdb.set_trace()
            word_count = len(abstract.split())
            if 150 <= word_count <= 250:
                # Add to the filtered data dictionary
                filtered_data[application_number] = {
                    "title": title,
                    "abstract": abstract
                }

# Print the total number of entries
print(f'Total number of entries with abstract length between 150 to 250 words: {len(filtered_data)}')


# Randomly sample 2000 entries if there are enough
if len(filtered_data) > 2000:
    sampled_data = dict(random.sample(filtered_data.items(), 2000))
else:
    sampled_data = filtered_data  # Use all data if less than 2000

# Print the total number of entries sampled
print(f'Total number of sampled entries: {len(sampled_data)}')

# Path for the output JSON file
output_file_path = './data_new/patent.json'

# Save the sampled data to a new JSON file
# with open(output_file_path, 'w') as outfile:
#     json.dump(sampled_data, outfile, indent=4)
# Calculate word counts for titles and abstracts
title_word_counts = [len(data['title'].split()) for data in sampled_data.values()]
abstract_word_counts = [len(data['abstract'].split()) for data in sampled_data.values()]

# Compute mean and standard deviation for titles
title_mean = np.mean(title_word_counts)
title_std = np.std(title_word_counts)

# Compute mean and standard deviation for abstracts
abstract_mean = np.mean(abstract_word_counts)
abstract_std = np.std(abstract_word_counts)

print(f"Mean word count for titles: {title_mean} ± {title_std}")
print(f"Mean word count for abstracts: {abstract_mean} ± {abstract_std}")