# -*- coding: utf-8 -*-

import os
import pandas as pd
import docx2txt
import re
import locale
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Switch to German format for date
locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
locale.setlocale(locale.LC_ALL, "deu_deu")

# NOTE: Due to copyright laws, we cannot publicly publish our gathered newspaper articles.

# Path to files with articles gathered through websearch
PATH_WEB = r"..."

# Path to files with articles gathered through LexisNexis
PATH_LEXISNEXIS = r"..."

# Path to files with articles gathered through the Ostseezeitung archive
PATH_SZ = r"..."

# Path to files with articles gathered through the Süddeutsche Zeitung archive
PATH_OZ = r"..."

def load_fishery_data(date = "2009-01-01"):
    
    """
    Load and combine fishery-related articles from various sources.

    Returns:
        pd.DataFrame: Combined data from Websearch, LexisNexis, Süddeutsche Zeitung, and Ostsee Zeitung.
    """
    
    dates = []
    filtered_text = []
    name = []
    headings = []
    fischerei_flat = pd.DataFrame()
    id_extra = pd.DataFrame()
    
    ###
    # LexisNexis
    ###
    
    for folder in os.listdir(PATH_LEXISNEXIS)[1:]:    
        
        folder_path = os.path.join(PATH_LEXISNEXIS, str(folder))

        # Load raw articles from LexisNexis
        for file in os.listdir(folder_path): 
            
            file_path = os.path.join(folder_path, file)
            text = docx2txt.process(file_path)
            
            first, *second = text.split('Body', 1)
                     
            # Process the headings
            head = ["".join(x) for x in re.findall(r'Page \d{1,3} of \d{1,3}\n\n(.*?)\n|Page  of \n\n(.*?)\n', first, re.M)]
            count = list(map(int, re.findall(r'Page (.*?) of', first, re.M)))
            if folder == '2021':
                head = head[1:]
                head.pop(11)
            head_count = list(zip(count, head))
            head = [x for _, x in sorted(head_count)]
            headings.append(head)

            try:
                second =  text.split('Body',1)[1]
                second = 'Body' + second
            except:
                second = ""
            
            # Load the main texts from Worddocs
            first = first.replace('\n', '')
            second = second.replace('\n', '')
            articles_first = re.sub(r'Body(.*?)Classification', '', first)
            articles_second = re.sub(r'Body(.*?)Classification', '' ,second)
            dates.append(re.findall("\d{1,2}\. \w+ \d{4}", articles_first)[1:] + re.findall("\d{1,2}\. \w+ \d{4}",  articles_second))    
        
            name.append(str(file))
            filtered_text.append(re.findall(r'Body(.*?)Classification',second))
            
        if folder == '2001':
            headings[1] = headings[1][1:]
    
    # Flatten the lists
    filtered_text_flat = [article for articles in filtered_text for article in articles]
    dates_flat = [d for ds in dates for d in ds]
    headings_flat = [h for hs in headings for h in hs]
    
    headings_flat.remove('No Headline In Original1')
    headings_flat.remove('No Headline In Original2')
    
    # Store extra information
    for j in range(2004, 2022):
        extra_info = pd.read_excel(os.path.join(PATH_WEB, 'sentiment analysis, newspaper, 10022023 FINAL.xlsx'), sheet_name=str(j))
        extra_info = extra_info.dropna(subset=['id', 'Heading']).reset_index()
        id_extra = pd.concat([id_extra, extra_info[[extra_info['Heading'][i] in headings_flat for i in range(len(extra_info))]]])

    id_extra = id_extra[['Heading', 'id', 'type of newspaper (regional, national)', 'Journal', 'Klima', 'Naturschutz', 'Fischerei']] 
    fischerei_flat = pd.DataFrame({'main_articles': filtered_text_flat, 'Date':dates_flat, 'Heading': headings_flat})
    fischerei_flat = pd.merge(fischerei_flat, id_extra, on = 'Heading', how="inner")
    fischerei_flat.dropna(subset = ['id'], inplace = True)
    fischerei_flat.drop_duplicates(subset = ['id'], inplace = True)
    
    ###
    # Süddeutsche Zeitung
    ###
    
    word_data = pd.DataFrame()
    
    # Load the main texts
    for j in range(2008,2022):
        articles = docx2txt.process(PATH_WEB + '\\' + str(j) + '\\' + 'np.' + str(j)+", all, txt.docx")  
        extra_info = pd.read_excel(PATH_WEB + "\\" + 'sentiment analysis, newspaper, 10022023 FINAL.xlsx', sheet_name=str(j))
        extra_info = extra_info.dropna(subset = ['id']).reset_index()
        extra_info = extra_info.dropna(subset=['Heading'])
        extra_info = extra_info.reset_index()
        
        if j <= 2016 or j == 2020 or j == 2014 or j == 2020 or j == 2021:
            ids_art = re.findall(str(j)+".np.\d{1,3}",articles)
            main_articles = [i.split("\nMT\n")[1] for i in re.split(str(j)+".np.\d{1,3}", articles)[1:]]
                      
        else:
            ids_art = re.findall("np."+str(j)+"\.\d{1,3}",articles)
            
            main_articles = [i.split("\nMT\n")[1] for i in re.split("np."+str(j)+".\\d{1,3}", articles)[1:]]

        year_data = pd.DataFrame({'id': ids_art,'main_articles': main_articles})
        word_data = pd.concat([word_data, pd.merge(extra_info, year_data, on=['id'], how='inner')])
    
    word_data = word_data.reset_index(drop = True)
    word_data = word_data[word_data['Reference'] != 'https://archiv.szarchiv.de/']
    
    # Store extra information
    extra_info_SZ = pd.DataFrame()
    
    for year in range(2004, 2022):
        try:
            extra_info_SZ = pd.concat([extra_info_SZ, pd.read_excel(PATH_WEB + "\\" + 'sentiment analysis, newspaper, 10022023 FINAL.xlsx', sheet_name=str(year))])
        except ValueError:
            continue
    
    extra_info_SZ = extra_info_SZ.fillna('').reset_index(drop=True)
    SZ_articles = extra_info_SZ[extra_info_SZ['Journal'] == 'Süddeutsche Zeitung']
    SZ_articles.loc[:, 'Heading'] = SZ_articles.loc[:, 'Heading'].str.replace('\n', '')
    
    text = docx2txt.process(PATH_SZ + '\\'+ 'SZ Articles.docx')
    text = text.replace('\n', '')
    texts_SZ = re.findall(r'Bodyt(.*?)Bodyt', text)
    headings_SZ = re.findall(r'Heading(.*?)Heading', text)
    
    headings_dict = {headings_SZ[i]: texts_SZ[i] for i in range(len(headings_SZ)) if headings_SZ[i] in SZ_articles['Heading'].values}
    
    data_SZ = SZ_articles[SZ_articles['Heading'].isin(headings_dict.keys())].copy()
    data_SZ.loc[:, 'main_articles'] = data_SZ['Heading'].apply(lambda x: headings_dict[x])
   
    ###
    # Ostsee Zeitung
    ###
    
    # Store extra information
    sentiment_analysis_data = {}
    for year in [2011, 2012, 2014, 2015, 2016, 2017, 2019, 2020, 2021]:
        extra_info_oz = pd.read_excel(PATH_WEB + "\\" + 'sentiment analysis, newspaper, 10022023 FINAL.xlsx', sheet_name=str(year))
        extra_info_oz = extra_info_oz.fillna('').reset_index(drop=True)
        extra_info_oz = extra_info_oz[extra_info_oz['Journal'] == 'OstseeZeitung']
        extra_info_oz.loc[:,'Heading'] = extra_info_oz.loc[:,'Heading'].str.replace('\n', '')
        sentiment_analysis_data[year] = extra_info_oz.reset_index(drop=True)
    
    full_data_oz = []
    
    # Load the main texts 
    for year in sentiment_analysis_data.keys():
        
        text = docx2txt.process(PATH_OZ +'\\'+ 'OstseeZeitung_' + str(year) + '.docx')
        
        if year in [2011, 2012, 2014, 2015, 2016, 2017]:
            text = text.replace('\n', '')
            text_oz = re.findall(r'Bodyt(.*?)Bodyt', text)
            headings_oz = re.findall(r'Heading(.*?)Heading', text)
            
            data_oz = sentiment_analysis_data[year][sentiment_analysis_data[year]['Heading'].isin(headings_oz)].copy()
            data_oz.loc[:,'main_articles'] = data_oz['Heading'].apply(lambda x: text_oz[headings_oz.index(x)] if x in headings_oz else '')
    
        else:
            pattern = f"np.{year}\\.\\d{{1,3}}" if year != 2020 else f"{year}\\.np\\.\\d{{1,3}}"
            text_oz = [i.split("\nMT\n")[1] for i in re.split(pattern, text)[1:]]
            ids_art = re.findall(pattern, text)
            
            year_data = pd.DataFrame({'id': ids_art,'main_articles':text_oz})
            data_oz = pd.merge(sentiment_analysis_data[year], year_data, on=['id'], how='inner')
    
        full_data_oz.append(data_oz)
    
    full_data_oz = pd.concat(full_data_oz)
    
    fischerei_flat['Date'] = pd.to_datetime(fischerei_flat['Date'], format='%d. %B %Y')
    
    # Combine all articles into one dataframe
    full_data = pd.concat([word_data, fischerei_flat, data_SZ, full_data_oz], sort=False)
    full_data.dropna(subset=['id'], inplace=True)
    full_data.drop_duplicates(subset=['id'], inplace=True)
    full_data.reset_index(drop=True, inplace=True)
    
    full_data.replace({np.nan: '', 0: ''}, regex=True, inplace=True)
    full_data['main_articles'] = full_data['Heading'] + full_data['subdescription'] + full_data['main_articles']
    full_data['main_articles'] = full_data['main_articles'].replace('\sBU\s', '', regex=True)
    
    # Only keep articles from journals from which at least 3 articles appear in our dataset
    number_journals = pd.read_excel("Data\journal_occur_small_journals_ordered.xlsx", header = None)   
    number_journals.columns = ['journal', 'publications','keep']
    min_number_journals = number_journals[number_journals['publications'] < 3][['journal','keep']]
    min_number_journals = min_number_journals[min_number_journals['keep'] == 0]['journal'].tolist()

    filtered_data = full_data[~full_data['Journal'].isin(min_number_journals)]
    
    # Delete Swiss newspaper articles
    filtered_data = filtered_data[~(filtered_data['Journal'] == 'SDA')] 
    
    # Filter out newspapers with missing permit
    filtered_data = filtered_data[~(filtered_data['Journal'] == 'NDR')] 
    filtered_data = filtered_data[~(filtered_data['Journal'] == 'Onvista')] 
    filtered_data = filtered_data[~(filtered_data['Journal'] == 'RTL')] 
    filtered_data = filtered_data[~(filtered_data['Journal'] == 'T-Online')] 
    filtered_data = filtered_data[~(filtered_data['Journal'] == 'Yahoo')] 
    filtered_data = filtered_data[~(filtered_data['Journal'] == 'CDU/CSU')] 
    
    filtered_data = filtered_data[filtered_data['Date'] >= date]
    
    return(filtered_data)

def clean_data(data):
    
    data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
    data['year'] = data['Date'].dt.year
    
    data['text'] = (data['text'].replace(';', '.', regex=True)
                                .str.replace('\.\.', '.', regex=True)
                                .str.replace('\. \.', '.', regex=True)
                                .str.replace('u ̈ ', 'ü', regex=False)
                                .str.replace('o ̈ ', 'ü', regex=False)
                                .str.replace('a ̈ ', 'ä', regex=False))

    return data