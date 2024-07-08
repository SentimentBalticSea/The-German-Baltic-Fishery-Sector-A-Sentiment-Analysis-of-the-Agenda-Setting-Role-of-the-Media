# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from count_stakeholders import stakeholder_pers, stakeholder_org
from data_prep import sentiment_index_yearly, fishery_year

def stakeholder_analysis(data, common_pers, common_orgs, stakeholders_pers_list, stakeholders_orgs_list, start_year, end_year, quota_dates):
    
    """
    Perform stakeholder analysis on the dataset.
    
    Args:
        data (pd.DataFrame): The main data containing articles and metadata.
        common_pers (pd.DataFrame): DataFrame containing names of all stakeholders (individuals).
        common_orgs (pd.DataFrame): DataFrame containing names of all stakeholders (organizations).
        stakeholders_pers_list (list): List of selected stakeholder individuals' names to analyze.
        stakeholders_orgs_list (list): List of selected stakeholder organizations' names to analyze.
        start_year (int): The starting year of the analysis.
        end_year (int): The ending year of the analysis.
        quota_dates (list): List of dates marking quota changes.
    
    Returns:
        Various DataFrames containing analysis results for stakeholders (individuals and organizations).
    """
    
    def filter_stakeholders(data, common_pers, common_orgs, stakeholders_pers_list, stakeholders_orgs_list, start_year, end_year):
        """
        Filter and analyze stakeholders within the data.
        
        Args:
            data (pd.DataFrame): The main data containing articles and metadata.
            common_pers (pd.DataFrame): DataFrame containing names of all stakeholders (indiviuals)
            common_orgs (pd.DataFrame): DataFrame containing names of all stakeholders (organizations)
            stakeholders_pers_list (list): List of selected stakeholder indiuals names to analyze.
            stakeholders_orgs_list (list): List of selected stakeholder organization names to analyze.
            start_year (int): The starting year of the analysis.
            end_year (int): The ending year of the analysis.
        
        Returns:
            stakeholders_pers_filtered (pd.DataFrame): DataFrame containing the number of occurences of selected stakeholders (individuals)
            stakeholders_orgs_filtered (pd.DataFrame): DataFrame containing the number of occurences of selteced stakeholders (organizations)
            stakeholders_data_pers (pd.DataFrame): DataFrame containing all sentences with selected stakeholders (indiviuals)
            stakeholders_data_orgs (pd.DataFrame): DataFrame containing all sentences with selected stakeholders (organizations)
        """

        def filter_stakelist(stakeholders_df, stakeholders_list):
            return stakeholders_df[stakeholders_df['Name'].str.lower().isin(stakeholders_list)]
        
        # Sort data into 'fishery years' based on the dates quota changes
        year_data = fishery_year(data, quota_dates)
        
        # Get all sentences which contain the name of a stakeholder and count how often a stakeholder is mentioned
        stakeholders_pers, stakeholders_data_pers, id_pers, text_pers = stakeholder_pers(year_data, common_pers, stakeholders_pers_list, start_year, end_year, quota_dates)
        stakeholders_data_pers.reset_index(inplace=True, drop=True)
        
        stakeholders_orgs, stakeholders_data_orgs, id_org, text_org = stakeholder_org(year_data, common_orgs, stakeholders_orgs_list, start_year, end_year, quota_dates)
        stakeholders_data_orgs.reset_index(inplace=True, drop=True)
        
        stakeholders_data_pers.drop_duplicates(['lemmas'],inplace=True)
        stakeholders_data_orgs.drop_duplicates(['lemmas'],inplace=True)
        
        # Count number of stakeholder occurences (only once per article)
        stakeholders_pers['Value_once'] = [len(set(id_list)) for id_list in id_pers['ID']] 
        stakeholders_orgs['Value_once'] = [len(set(id_list)) for id_list in id_org['ID']] 

        # Filter out all selected stakeholders 
        stakeholders_pers_filtered = filter_stakelist(stakeholders_pers, [name.lower() for name in stakeholders_pers_list])
        stakeholders_orgs_filtered = filter_stakelist(stakeholders_orgs, [name.lower() for name in stakeholders_orgs_list])      
                   
        return(stakeholders_pers_filtered, stakeholders_data_pers, stakeholders_orgs_filtered, stakeholders_data_orgs)
    
    data = data[(data['Date'].dt.year >= start_year) & (data['Date'].dt.year <= end_year)]
    
    stakeholders_pers_filtered, stakeholders_data_pers, stakeholders_orgs_filtered, stakeholders_data_orgs = filter_stakeholders(data, common_pers, common_orgs, stakeholders_pers_list, stakeholders_orgs_list, start_year, end_year)
    
    def stakeholder_share(data, stakeholders_pers_filtered, stakeholders_orgs_filtered):
        """
        Calculate the share of stakeholders per articles in the dataset.
        
        Args:
            data (pd.DataFrame): The main data containing articles and metadata.
            stakeholders_pers_filtered (pd.DataFrame): DataFrame containing the number of occurences of selected stakeholders (individuals)
            stakeholders_orgs_filtered (pd.DataFrame): DataFrame containing the number of occurences of selteced stakeholders (organizations)
            quota_dates (list): List of dates marking quota changes.
        
        Returns:
            stakeholders_pers_sum (pd.DataFrame): DataFrame with calculated shares per articles for stakeholders (indivuals).
            stakeholders_orgs_sum (pd.DataFrame): DataFrame with calculated shares per articles for stakeholders (organizations).
            count_results_list (list): List of counts of articles per year.
            stakeholders_pers_sum_all_persons (pd.DataFrame): DataFrame with yearly summed up stakeholders (indivuals).
            stakeholders_orgs_sum_all_orgs (pd.DataFrame): DataFrame with yearly summed up stakeholders (organizations).
        """
         
        # Count number of stakeholders per year
        if stakeholders_pers_filtered.empty:
            stakeholders_pers_sum = pd.DataFrame({'Year': pd.to_datetime(quota_dates[:-1]), 'Value': [0]*len(quota_dates[:-1])})
            stakeholders_pers_sum_all_persons = []
            stakeholders_pers_sum_once = pd.DataFrame({'Year': pd.to_datetime(quota_dates[:-1]), 'Value_once': [0]*len(quota_dates[:-1])})
        else:
            stakeholders_pers_sum = stakeholders_pers_filtered[['Year', 'Value']].groupby(['Year']).sum()
            stakeholders_pers_sum_all_persons = stakeholders_pers_filtered[['Name', 'Value']].groupby(['Name']).sum()
            stakeholders_pers_sum_once = stakeholders_pers_filtered[['Year', 'Value_once']].groupby(['Year']).sum()
            
        if stakeholders_orgs_filtered.empty:
            stakeholders_orgs_sum = pd.DataFrame({'Year': pd.to_datetime(quota_dates[:-1]), 'Value': [0]*len(quota_dates[:-1])})
            stakeholders_orgs_sum_all_orgs = []
            stakeholders_orgs_sum_once = pd.DataFrame({'Year': pd.to_datetime(quota_dates[:-1]), 'Value': [0]*len(quota_dates[:-1])})
        else:
            stakeholders_orgs_sum = stakeholders_orgs_filtered[['Year', 'Value']].groupby(['Year']).sum()
            stakeholders_orgs_sum_all_orgs = stakeholders_orgs_filtered[['Name', 'Value']].groupby(['Name']).sum()
            stakeholders_orgs_sum_once = stakeholders_orgs_filtered[['Year', 'Value_once']].groupby(['Year']).sum()
                  
        data_sorted = fishery_year(data, quota_dates)
        data_sorted = [df for df in data_sorted if not df.empty]
      
        # Count number of articles per year
        count_results_list = []  
      
        for yearly_data in data_sorted:

            count_results = len(set(yearly_data['id']))
            count_results_list.append(count_results)

        # Calculate share of stakeholders in relation to the total number of articles per year
        stakeholders_pers_sum['share'] = np.array(stakeholders_pers_sum['Value'])/np.array(count_results_list)
        stakeholders_pers_sum['share_once'] = np.array(stakeholders_pers_sum_once['Value_once'])/np.array(count_results_list)
        
        stakeholders_orgs_sum['share'] = np.array(stakeholders_orgs_sum['Value'])/np.array(count_results_list)
        stakeholders_orgs_sum['share_once'] = np.array(stakeholders_orgs_sum_once['Value_once'])/np.array(count_results_list)
        
        return(stakeholders_pers_sum, stakeholders_orgs_sum, count_results_list, stakeholders_pers_sum_all_persons, stakeholders_orgs_sum_all_orgs)
    
    stakeholders_pers_sum, stakeholders_orgs_sum, count_results_list, stakeholders_pers_sum_all_persons, stakeholders_orgs_sum_all_orgs = stakeholder_share(data, stakeholders_pers_filtered, stakeholders_orgs_filtered)   
    
    def stakeholder_sentiment(stakeholders_data_pers, stakeholders_data_orgs, quota_dates):
        """
        Calculate the sentiment index for stakeholders.
    
        Args:
            stakeholders_data_pers (pd.DataFrame): DataFrame containing data for person stakeholders.
            stakeholders_data_orgs (pd.DataFrame): DataFrame containing data for organization stakeholders.
            quota_dates (list): List of dates marking quota changes.
    
        Returns:
            stakeholders_pers_sentiment (pd.DataFrame): DataFrame with calculated sentiment index for person stakeholders.
            stakeholders_orgs_sentiment (pd.DataFrame): DataFrame with calculated sentiment index for organization stakeholders.
        """
        
        # Sort data into 'fishery years' based on the dates quota changes
        fishery_year_data = fishery_year(data, quota_dates)
        sentiment = sentiment_index_yearly(fishery_year_data)
        sentiment['n_sents'] = sentiment['pos'] + sentiment['neu'] +sentiment['neg']
        
        # Calculate yearly sentiment index for sentences which include selected stakeholders (individuals)
        if stakeholders_data_pers.empty:
            fishery_year_stakeholders_pers = pd.DataFrame()
            stakeholders_pers_sentiment = 0  
        else:
            fishery_year_stakeholders_pers = fishery_year(stakeholders_data_pers, quota_dates)
            stakeholders_pers_sentiment = sentiment_index_yearly(fishery_year_stakeholders_pers)
            stakeholders_pers_sentiment['n_sents'] = stakeholders_pers_sentiment['pos'] + stakeholders_pers_sentiment['neu'] + stakeholders_pers_sentiment['neg']
        
        # Calculate yearly sentiment index for sentences which include selected stakeholders (organizations)
        if stakeholders_data_orgs.empty:
            
            fishery_year_stakeholders_orgs = pd.DataFrame()
            stakeholders_orgs_sentiment = 0
        else:
            fishery_year_stakeholders_orgs = fishery_year(stakeholders_data_orgs, quota_dates)
            stakeholders_orgs_sentiment = sentiment_index_yearly(fishery_year_stakeholders_orgs)
            stakeholders_orgs_sentiment['n_sents'] = stakeholders_orgs_sentiment['pos'] + stakeholders_orgs_sentiment['neu'] + stakeholders_orgs_sentiment['neg']
        
        return(stakeholders_pers_sentiment, stakeholders_orgs_sentiment)
    
    sentiment_pers_stakeholders, stakeholders_orgs_sentiment = stakeholder_sentiment(stakeholders_data_pers, stakeholders_data_orgs, quota_dates)
    
    def stakeholders_both(data, quota_dates, stakeholders_data_pers, stakeholders_data_orgs, stakeholders_orgs_filtered, stakeholders_pers_filtered):
        """
        Combine person and organization stakeholders into one group and calculate their shares and sentiments.
        
        Args:
            data (pd.DataFrame): The main data containing articles and metadata.
            quota_dates (list): List of dates marking quota changes.
            stakeholders_data_pers (pd.DataFrame): DataFrame containing all sentences with selected stakeholders (indiviuals)
            stakeholders_data_orgs (pd.DataFrame): DataFrame containing all sentences with selected stakeholders (organizations)
            stakeholders_orgs_filtered (pd.DataFrame): DataFrame containing the number of occurences of selected stakeholders (organizations)
            stakeholders_pers_filtered (pd.DataFrame): DataFrame containing the number of occurences of selected stakeholders (individuals)
        
        Returns:
            DataFrames with combined stakeholder metrics.
        """
        
        data.loc[:, 'year'] = data['Date'].dt.year
        
        # Combine persons and organisations to one stakeholder group
        stakeholders_data_both = pd.concat([stakeholders_data_pers, stakeholders_data_orgs])
        
        stakeholders_data_both.reset_index(inplace = True, drop = True)
        
        stakeholders_both_list = pd.concat([stakeholders_orgs_filtered, stakeholders_pers_filtered])
        stakeholders_both_sum = stakeholders_both_list[['Year', 'Value']].groupby(['Year']).sum()
        stakeholders_both_sum['share'] = np.array(stakeholders_both_sum['Value'])/np.array(count_results_list)
        
        stakeholders_both_sum_once = stakeholders_both_list[['Year', 'Value_once']].groupby(['Year']).sum()
        stakeholders_both_sum['share_once'] = np.array(stakeholders_both_sum_once['Value_once'])/np.array(count_results_list)
        
        fishery_year_stakeholders_both = fishery_year(stakeholders_data_both, quota_dates)
        stakeholders_sentiment_both = sentiment_index_yearly(fishery_year_stakeholders_both)
        stakeholders_sentiment_both['n_sents'] = stakeholders_sentiment_both['pos'] + stakeholders_sentiment_both['neu'] + stakeholders_sentiment_both['neg']
               
        return(stakeholders_both_sum, stakeholders_sentiment_both)
    
    stakeholders_both_sum, stakeholders_sentiment_both = stakeholders_both(data, quota_dates, stakeholders_data_pers, stakeholders_data_orgs, stakeholders_orgs_filtered, stakeholders_pers_filtered)
    
    return(stakeholders_pers_sum, sentiment_pers_stakeholders, stakeholders_orgs_sum, stakeholders_orgs_sentiment, stakeholders_both_sum, stakeholders_sentiment_both, stakeholders_data_orgs, stakeholders_data_pers, count_results_list, stakeholders_pers_sum_all_persons, stakeholders_orgs_sum_all_orgs)