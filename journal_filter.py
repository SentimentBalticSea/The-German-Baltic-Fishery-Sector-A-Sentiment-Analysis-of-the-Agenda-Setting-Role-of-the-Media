# -*- coding: utf-8 -*-

import pandas as pd
from load_data import load_fishery_data
import os

# Load data
data = load_fishery_data()
data = data.rename(columns={'main_articles': 'text'})
data.drop(['level_0', 'index', 'Supporting information (0=no, 1=yes)', '#',
           'Content', 'Autor', 'Heading', 'subdescription',
           'number of pics', 'content of pics', 'videos (0=no, 1=yes)',
           'number of video', 'content of video', 'Reference',
           'photograph (0=no, 1=yes)', 'Number of SI',
           'Bermerkung', 'Bemerkung'], axis=1, inplace=True)

data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
data['year'] = data['Date'].dt.year

data = data[data["Date"] >= "2009-01-01"]

# Create a mapping of lowercased journal names to their original format
journal_names_map = data['Journal'].dropna().unique()
journal_names_map = {name.lower(): name for name in journal_names_map}

# Count how often each journal occurs in the dataset
data['Journal'] = data['Journal'].str.strip().str.lower()
journal_occur = data.groupby('Journal').size()

# Filter journals with at least 3 occurrences
journals_to_keep = journal_occur[journal_occur >= 3].index
filtered_data = data[data['Journal'].isin(journals_to_keep)]

filtered_journal_occur = filtered_data.groupby('Journal').size().reset_index(name='Occurrences')
filtered_journal_occur = filtered_journal_occur.sort_values(by='Occurrences', ascending=False).reset_index(drop=True)

# Count number of regional newspapers
sum(filtered_data['type of newspaper (regional, national)'] == 'regional')

# Count number of articles from dpa
pattern = r'\(dpa\)|\(dpa/lno\)|\(dpa/mv\)|\(dpa-AFX\)'
dpa_articles = filtered_data[filtered_data['text'].str.contains(pattern, na=False)]

# Calculate share of dpa articles
len(dpa_articles)/len(filtered_data)

filtered_journal_occur['Journal'] = filtered_journal_occur['Journal'].map(journal_names_map)

# Save filtered data
filtered_data.to_excel("Data/filtered_data.xlsx")

# Generate Excel file for Table S3
output_path = os.path.join('Results', 'monthly_article_means_fig2a.xlsx')
filtered_journal_occur.to_excel(output_path, index=False)

