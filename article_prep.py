# -*- coding: utf-8 -*-

import pandas as pd
import count_words
from tqdm import tqdm

from data_prep import tokenizer_articles, lemmatize_sentences

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
data = pd.read_excel("filtered_data.xlsx")
data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
data['year'] = data['Date'].dt.year
data.reset_index(inplace = True, drop = True)  

def prepare_articles(data, language = "german"): 
    """
    Prepares articles by tokenizing, lemmatizing, and splitting into sentences.
    
    Args:
        data (pd.DataFrame): DataFrame containing articles.
        language (str): Language for processing (default is "german").
    
    Returns:
        pd.DataFrame: DataFrame with processed sentences.
        pd.DataFrame: Original DataFrame with additional lemma column.
    """
    data['text'] = data['text'].replace('BU', '', regex = True) 
    
    data['text'] = tokenizer_articles(data, language = "german")['text']
    
    # Lemmatize texts
    data['Lemmas'] = [lemmatize_sentences(text, language = "german") for text in tqdm(data['text'], desc="Lemmatizing texts")] 
    
    data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d') 
    data['year'] = data['Date'].dt.year 
    data.dropna(subset = ['Lemmas'], inplace = True) 
    
    # Split articles into sentences
    sentence_split = data.explode('text') 
    
    combined = data.apply(lambda row: list(zip(row['text'], row['Lemmas'])), axis=1)
    combined_split = combined.explode().apply(pd.Series)
    combined_split.columns = ['text', 'Lemmas']
    
    sentence_split['Lemmas'] = combined_split['Lemmas']
    
    # Filter sentences
    unwanted_phrases = ['Foto :', 'Graphic']
    sentence_split = sentence_split[~sentence_split['text'].str.contains('|'.join(unwanted_phrases))]
    sentence_split = sentence_split[sentence_split['text'].str.strip() != '']
            
    # Count words
    sentence_split['word_count'] = [count_words.count_words(text) for text in tqdm(sentence_split['text'], desc="Counting words")] 
    
    sentence_split['text'] = sentence_split['text'].str.replace('\.\.', '.', regex = True).str.replace('\. \.', '.', regex = True)
    
    # Filter sentences with at least 4 words
    sentence_split = sentence_split[sentence_split['word_count'] >= 4]
    sentence_split.reset_index(inplace=True, drop=True) 
 
    return sentence_split, data

data_sentences, data = prepare_articles(data)

# Keywords for identifying articles about herring and cod
hering_keywords = ['hering']
cod_keywords = ['dorsch', 'kabeljau']

# Separate articles about cod or herring from the corpus
hering_articles_idx = []
cod_articles_idx = []

# seperate articles about cod or hering from the corpus
for idx, article in enumerate(data['Lemmas']):
    
    article = ' '.join(article)
    if any(word in article for word in hering_keywords):
        hering_articles_idx.append(idx)
        
    if any(word in article for word in cod_keywords):
        cod_articles_idx.append(idx)

# Get herring and cod articles
hering_articles = data.iloc[hering_articles_idx].reset_index(drop = True)
cod_articles = data.iloc[cod_articles_idx].reset_index(drop = True)

# Get sentences related to herring and cod
hering_sentences = data_sentences[data_sentences['id'].isin(hering_articles['id'])]
cod_sentences = data_sentences[data_sentences['id'].isin(cod_articles['id'])]

###############################################################################
# Save processed data to CSV files
###############################################################################
 
hering_articles.to_csv(r'fishery_lemmas_hering_articles.csv') 
hering_sentences.to_csv(r'fishery_lemmas_hering_sentences.csv') 
 
cod_articles.to_csv(r'fishery_lemmas_cod_articles.csv') 
cod_sentences.to_csv(r'fishery_lemmas_cod_sentences.csv') 

data.to_csv(r'fishery_lemmas_articles.csv') 
data_sentences.to_csv(r'fishery_lemmas_sentences.csv')

###############################################################################