# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import numpy as np

from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

# Set up path
BASE_PATH = r"Data"

tagger = SequenceTagger.load("flair/ner-german-large")
    
tags = []

# load data
data_art = pd.read_csv(r'Data\fishery_lemmas_articles.csv') 
data_art["Date"] = pd.to_datetime(data_art["Date"], format='%Y-%m-%d')

data_art['text'].replace(';','.', inplace = True, regex = True)
data_art['text'].replace('u ̈ ','ü', inplace = True, regex = True)
data_art['text'].replace('\u2028', ' ', inplace = True, regex = True)
data_art['text'].replace('\u0308 ', 'ü', inplace = True, regex = True)

# Split data into chunks for processing
data_splits = np.array_split(data_art['text'], 20)

# Process each chunk and extract NER tags
for split_idx, dat in enumerate(tqdm(data_splits, desc="Processing splits")):
    for text_idx, text in enumerate(tqdm(dat, desc=f"Processing texts in split {split_idx}")):
        # Clean individual text
        text = text.replace('\u2028', ' ')
        text = re.sub(r'u \u0308 ', 'ü', text)
        
        # Create sentence object for prediction
        sent = Sentence(text)
        tagger.predict(sent)
        
        # Extract tags
        tag = [(entity.text, entity.tag, entity.score) for entity in sent.get_spans('ner')]
        tags.append(tag)

# Add tags to dataframe
data_art['Tags'] = tags

# Flatten and process tags
flat_tags = [t for sublist in tags for t in sublist]
entity_df = pd.DataFrame(flat_tags, columns=['Name', 'Tag', 'Score'])
entity_df = entity_df[entity_df['Tag'] != 'LOC']
entity_occur = entity_df.groupby('Name').size()
entity_df = entity_df.groupby(['Name', 'Tag']).size().unstack(fill_value=0)

# Save entities to Excel
entities_excel_path = os.path.join(BASE_PATH, 'entities_art.xlsx')
entity_df.to_excel(entities_excel_path)

# The "entities_art" file was manually searched for potential
# organizational and individual stakeholders in the next step. 
# The results of this search are stored in the files "common_orgs" 
# for organizational stakeholders and "common_per" for individual stakeholders.