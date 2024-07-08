# -*- coding: utf-8 -*-

import pandas as pd
import nltk
import re
import shlex

def stakeholder_pers(data, common_pers, stakeholders, start_year, end_year, quota_dates):
    """
    Identify and count occurrences of person stakeholders in the dataset.
    
    Args:
        data (list of pd.DataFrame): Yearly data containing articles and metadata.
        common_pers (pd.DataFrame): DataFrame containing names of all stakeholders (indiviuals)
        stakeholders (list): List of selected stakeholder indiuals names to analyze.
        start_year (int): The starting year of the analysis.
        end_year (int): The ending year of the analysis.
        quota_dates (list): List of dates marking quota changes.
    
    Returns:
        stakeholders_pers (pd.DataFrame): DataFrame with counts of person stakeholders by year.
        stakeholders_data (pd.DataFrame): DataFrame with all sentences containing selected stakeholders.
        id_pers_df (pd.DataFrame): DataFrame with IDs of articles containing person stakeholders.
        text_pers_df (pd.DataFrame): DataFrame with texts of articles containing person stakeholders.
    """
    
    # Create a list of full names in lowercase for comparison
    full_names = [pers.lower().strip() for pers in common_pers['Name']]
    
    # Create dictionary that link last names to full names
    names_ref = {}
    for name in full_names:
        last_name = name.split()[-1].lower()  # Ensure case consistency
        if last_name in names_ref:
            names_ref[last_name].append(name)
        else:
            names_ref[last_name] = [name]
    
    # Initialize DataFrames to store results
    stakeholders_pers = pd.DataFrame()
    stakeholders_data = pd.DataFrame()
    id_pers_df = pd.DataFrame()
    text_pers_df = pd.DataFrame()
    
    for year_idx, yearly_data in enumerate(data):
        # Aggregate lemmas by article ID
        articles = yearly_data.groupby('id').agg({
                'lemmas': 'sum',  
        }).reset_index()
        
        # Initialize dictionaries to store counts and IDs for the current year
        dict_pers = {name: 0 for name in full_names}
        id_pers = {name: [] for name in full_names}
        text_pers = {name: [] for name in full_names}
        
        for idx, lemmas in enumerate(yearly_data['lemmas']):
            article = articles[articles['id'] == list(yearly_data.iloc[[idx]]['id'])[0]]
            lemmas_lower = lemmas.lower()
            tokens = nltk.word_tokenize(lemmas_lower)  # Tokenize in lowercase
            
            # Check for each token if it is last name from names_red dict
            for i, token in enumerate(tokens):
                if token in names_ref:
                    # Check if the token is last name that occurs multiple times in names_red dict
                    if len(names_ref[token]) > 1:  
                        for idx, single_name in enumerate(names_ref[token]):
                            # Match full name to avoid ambiguity
                            full_name = names_ref[token][idx]
                            # If name is found in article and is part of selected stakeholders increase counter for name by one
                            if full_name in stakeholders and full_name in list(article['lemmas'])[0]:
                                dict_pers[full_name] += 1
                                stakeholders_data = pd.concat([stakeholders_data, yearly_data.iloc[[idx]]], ignore_index=True)
                                id_pers[full_name].extend(list(yearly_data.iloc[[idx]]['id']))
                                text_pers[full_name].extend(list(yearly_data.iloc[[idx]]['text']))     
                    elif len(names_ref[token]) == 1:  
                        full_name = names_ref[token][0]
                        # If name is found in article and is part of selected stakeholders increase counter for name by one
                        if full_name in stakeholders and full_name in list(article['lemmas'])[0]:
                            dict_pers[full_name] += 1
                            stakeholders_data = pd.concat([stakeholders_data, yearly_data.iloc[[idx]]], ignore_index=True)
                            id_pers[full_name].extend(list(yearly_data.iloc[[idx]]['id']))
                            text_pers[full_name].extend(list(yearly_data.iloc[[idx]]['text']))

        # Compile the counts into DataFrames
        stakeholders_pers = pd.concat([stakeholders_pers, pd.DataFrame({'Name': list(dict_pers.keys()), 'Value': list(dict_pers.values()), 'Year': quota_dates[year_idx]})])
        
        # Compile the IDs and texts into DataFrames
        id_pers_df = pd.concat([id_pers_df, pd.DataFrame({'Name': list(id_pers.keys()), 'ID': id_pers.values(), 'Year': quota_dates[year_idx]})])
        text_pers_df = pd.concat([text_pers_df, pd.DataFrame({'Name': list(text_pers.keys()), 'ID': text_pers.values(), 'Year': quota_dates[year_idx]})])
    
    # Remove duplicates from stakeholders_data
    stakeholders_data.drop_duplicates(inplace=True, ignore_index=True)

    return stakeholders_pers, stakeholders_data, id_pers_df, text_pers_df

def stakeholder_org(data, common_org, stakeholders, start_year, end_year, quota_dates):
    """
    Identify and count occurrences of organizational stakeholders in the dataset.
    
    Args:
        data (list of pd.DataFrame): Yearly data containing articles and metadata.
        common_org (pd.DataFrame): DataFrame containing names of all stakeholders (organizations).
        stakeholders (list): List of selected stakeholder organizations' names to analyze.
        start_year (int): The starting year of the analysis.
        end_year (int): The ending year of the analysis.
        quota_dates (list): List of dates marking quota changes.
    
    Returns:
        stakeholders_org (pd.DataFrame): DataFrame with counts of organizational stakeholders by year.
        stakeholders_data (pd.DataFrame): DataFrame with all sentences containing selected stakeholders.
        id_org_df (pd.DataFrame): DataFrame with IDs of articles containing organizational stakeholders.
        text_org_df (pd.DataFrame): DataFrame with texts of articles containing organizational stakeholders.
    """
    name_orgs = list(common_org['Name'])
    search_org = []
    
    id_org_df =  pd.DataFrame()
    text_org_df = pd.DataFrame()
    
    for i in range(0,5):
    
        common_org.iloc[:,i+6] = [' '.join(shlex.split(lemma, posix = False)) if lemma == lemma else '' for lemma in common_org.iloc[:,i+6]]
        search_org.extend([lemma.lower() for lemma in list(common_org.iloc[:,i+6]) if lemma != '']) 
    
    stakeholders_org = pd.DataFrame()
    stakeholders_data = pd.DataFrame()
    
    for year_idx, yearly_data in enumerate(data):
        
        dict_org = dict(zip(name_orgs, [0]*len(name_orgs)))
        id_org = {name: [] for name in name_orgs}
        text_org = {name: [] for name in name_orgs}
        
        for i, lemmas in enumerate(yearly_data['lemmas']):
        
            for idx, alternatives in enumerate(zip(common_org['Name'], common_org['Lemma1'], common_org['Lemma2'], common_org['Lemma3'], common_org['Lemma4'], common_org['Lemma5'],common_org['Abkürzung'])):
                
                        def count_orgs(alternatives, idx, name_orgs, dict_org, lemmas, stakeholders_data):
                            
                            ab_in_text = False
                            ab_tried = False
                            
                            added_indices = set()
                                
                            for orgs in alternatives[1:-1]:
                                
                                if orgs == orgs:
                                
                                    regex = orgs
                                    regex = regex.strip()
            
                                    if regex != '':
                                                 
                                        if alternatives[-1] == alternatives[-1] and ab_tried == False:
                                                
                                                ab = alternatives[-1].strip()
                                                
                                                if ab != '':
                                                
                                                    ab_regex = r'\b' + '(?<!'+ regex + '\s)' + ab  + r'\b' # oder + '(?![\w-])'
                                                    
                                                    ab_ocur = len(re.findall(ab_regex, lemmas))
                                                    
                                                    if ab_ocur > 0:
    
                                                        
                                                        if name_orgs[idx] in stakeholders:
                                                            
                                                            ab_in_text = True
                                                            
                                                            dict_org[name_orgs[idx]] += ab_ocur
                                                            id_org[name_orgs[idx]].extend(list(yearly_data.iloc[[i]]['id']))
                                                            text_org[name_orgs[idx]].extend(list(yearly_data.iloc[[i]]['text']))
                                                            
                                                            stakeholders_data = pd.concat([stakeholders_data, yearly_data.iloc[[i]]], ignore_index=True)
                                                            
                                                        ab_tried = True 
                                     
                                        if ab_in_text == False:                        
                                     
                                            regex = r'\b' + regex + r'\b'
                                            
                                            # specific regex for 'EU'
                                            if alternatives[0] == 'EU':
                                                
                                                regex = r'\b' + regex + '(?![\w-])' # oder r'\b'
                                             
                                            # # specific regex for 'Landesfischereiverband'
                                            if alternatives[0] == 'Landesfischereiverband':
                                                
                                                regex = r'\b' + regex + r'(?!\smecklenburg-vorpommern|\sschleswig-holstein)\b'
                                            
                                            regex = regex.strip()    
                                            
                                            ocur = len(re.findall(regex, lemmas))
                                    
                                            if ocur > 0: 
                                            
                                                if i not in added_indices and name_orgs[idx] in stakeholders:
                                                    
                                                        if ('marckwardt' in lemmas or 'marquardt' in lemmas or 'flindt' in lemmas or 'schmöde' in lemmas or 'schleswig-holstein' in lemmas) and name_orgs[idx] == 'Landesfischereiverband':
                                                            
                                                            dict_org['Landesfischereiverband Schleswig-Holstein'] += ocur
                                                            id_org['Landesfischereiverband Schleswig-Holstein'].extend(list(yearly_data.iloc[[i]]['id']))
                                                            text_org['Landesfischereiverband Schleswig-Holstein'].extend(list(yearly_data.iloc[[i]]['text']))
    
                                                            stakeholders_data = pd.concat([stakeholders_data, yearly_data.iloc[[i]]], ignore_index=True)
                                                            added_indices.add(i)
                                                            
                                                        elif ('kahlfuß' in lemmas or 'kahlfuss' in lemmas or 'paetsch' in lemmas or 'schlüter' in lemmas or 'bork'in lemmas or 'schütt' in lemmas or 'mecklenburg-vorpommern' in lemmas) and name_orgs[idx] == 'Landesfischereiverband':
                                                            
                                                            dict_org['Landesfischereiverbandes Mecklenburg-Vorpommern'] += ocur
                                                            id_org['Landesfischereiverbandes Mecklenburg-Vorpommern'].extend(list(yearly_data.iloc[[i]]['id']))
                                                            text_org['Landesfischereiverbandes Mecklenburg-Vorpommern'].extend(list(yearly_data.iloc[[i]]['text']))
    
                                                            stakeholders_data = pd.concat([stakeholders_data, yearly_data.iloc[[i]]], ignore_index=True)
                                                            added_indices.add(i)
                                                            
                                                        else:
                                                    
                                                            dict_org[name_orgs[idx]] += ocur
                                                            id_org[name_orgs[idx]].extend(list(yearly_data.iloc[[i]]['id']))
                                                            text_org[name_orgs[idx]].extend(list(yearly_data.iloc[[i]]['text']))
        
                                                            stakeholders_data = pd.concat([stakeholders_data, yearly_data.iloc[[i]]], ignore_index=True)
                                                            added_indices.add(i)

                            return(dict_org, stakeholders_data)
        
                        dict_org, stakeholders_data = count_orgs(alternatives, idx, name_orgs, dict_org, lemmas, stakeholders_data)                          
        
        stakeholders_org = pd.concat([stakeholders_org, pd.DataFrame({'Name': list(dict_org.keys()), 'Value': list(dict_org.values()), 'Year': quota_dates[year_idx]})])   
        
        id_org_df = pd.concat([id_org_df, pd.DataFrame({'Name': list(id_org.keys()), 'ID': id_org.values(), 'Year': quota_dates[year_idx]})])
        text_org_df = pd.concat([text_org_df, pd.DataFrame({'Name': list(text_org.keys()), 'Sentences': text_org.values(), 'Year': quota_dates[year_idx]})])
  
    stakeholders_data.drop_duplicates(inplace=True, ignore_index=False)

    return(stakeholders_org, stakeholders_data, id_org_df, text_org_df)