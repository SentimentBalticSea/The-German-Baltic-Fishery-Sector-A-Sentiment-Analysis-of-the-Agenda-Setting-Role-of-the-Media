# -*- coding: utf-8 -*-

import pandas as pd
import pylab as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from openpyxl import Workbook

plt.rcParams['figure.figsize'] = [11, 5]

from data_prep import fishery_year, sentiment_index_yearly, sentiment_index_quarterly

# Load all sentences
data_sents = pd.read_csv(r'Data\fishery_lemmas_sentences_labeled.csv')
data_sents["Date"] = pd.to_datetime(data_sents["Date"], format='%Y-%m-%d')

# Load all sentences from articles related to cod
data_sents_cod = pd.read_csv(r'Data\fishery_lemmas_cod_sentences.csv')
data_sents_cod["Date"] = pd.to_datetime(data_sents_cod["Date"], format='%Y-%m-%d')

# Load all sentences from articles related to herring
data_sents_herring = pd.read_csv(r'Data\fishery_lemmas_hering_sentences.csv')
data_sents_herring["Date"] = pd.to_datetime(data_sents_herring["Date"], format='%Y-%m-%d')

data_sents_cod = data_sents[data_sents['text'].isin(data_sents_cod['text'])]
data_sents_herring = data_sents[data_sents['text'].isin(data_sents_herring['text'])]

# Load quotes and dates of quota announcments
quota_dates = pd.read_excel('Data\data, fish & fisheries, SD22-24.xlsx', sheet_name = 'dates, advice - quota ')['quota decision']
quota_dates = quota_dates[:-1]
quota_dates = quota_dates[quota_dates >= "2007-01-01"]
quota_dates.reset_index(drop = True, inplace = True)

quota = pd.read_excel(r'Data\data, fish & fisheries, SD22-24.xlsx', sheet_name = 'dt, quota')
quota = quota[13:-1]
quota["Y"] = pd.to_datetime(quota["Y"], format='%Y')
quota = quota[quota["Y"] >= "2007-01-01"]
quota.reset_index(drop = True, inplace = True)

cod_quota = quota.iloc[:,[0,4]]
cod_quota['Unnamed: 4'] = quota.iloc[:,4].pct_change()
cod_quota['quota_abs'] = quota.iloc[:,4]
cod_quota['Y'] = quota_dates

cod_quota = cod_quota[2:]
cod_quota.iloc[:,1] = cod_quota.iloc[:,1]*100

herring_quota = quota.iloc[:,[0,2]]
herring_quota['Unnamed: 2'] = quota.iloc[:,2].pct_change()
herring_quota['quota_abs'] = quota.iloc[:,2]
herring_quota['Y'] = quota_dates

herring_quota = herring_quota[2:]
herring_quota.iloc[:,1] = herring_quota.iloc[:,1]*100

sentiment_all = sentiment_index_yearly(fishery_year(data_sents, quota_dates))
sentiment_all['Sentiment_change'] = sentiment_all["Sentiment Index"].pct_change()

sentiment_cod = sentiment_index_yearly(fishery_year(data_sents_cod, quota_dates))
sentiment_cod['Sentiment_change'] = sentiment_cod["Sentiment Index"].pct_change()

sentiment_herring = sentiment_index_yearly(fishery_year(data_sents_herring, quota_dates))
sentiment_herring['Sentiment_change'] = sentiment_herring["Sentiment Index"].pct_change()

sentiment_all_quarterly = sentiment_index_quarterly(data_sents, quota_dates)
sentiment_cod_quarterly = sentiment_index_quarterly(data_sents_cod, quota_dates)
sentiment_herring_quarterly = sentiment_index_quarterly(data_sents_herring, quota_dates)

sentiment_all['year'] = pd.to_datetime(sentiment_all['year'], format='%Y')
sentiment_cod['year'] = pd.to_datetime(sentiment_cod['year'], format='%Y')
sentiment_herring['year'] = pd.to_datetime(sentiment_herring['year'], format='%Y')

###############################################################################

# Calculate summary statistics  for sentimend indizes and create table S5 
def calculate_summary_stats(data):
    """Calculate summary statistics (mean, std, min, max) for the given data."""
    return data.describe().loc[['mean', 'std', 'min', 'max']]

def create_summary_statistics_excel(data, file_path):
    """
    Create an Excel file with summary statistics for sentiment indices.

    Args:
        data (dict): Dictionary of DataFrames containing sentiment indices.
        file_path (str): Path to save the Excel file.
    """
    summary_stats = {}
    for key, df in data.items():
        summary_stats[key] = calculate_summary_stats(df).round(3)
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary Statistics"
    ws.append([])
    ws.append(["Table S5. Sentiment indices by species"])
    ws.append([])

    # Quarterly data
    ws.append(["Quarterly"])
    ws.append(["", "Mean", "S.D.", "Min", "Max"])
    for group in ['Quarterly general sentiment', 'Quarterly herring sentiment', 'Quarterly cod sentiment']:
        stats = calculate_summary_stats(data[group])["Sentiment Index"]
        ws.append([group] + list(stats.values))

    # Yearly data
    ws.append([])
    ws.append(["Yearly"])
    ws.append(["", "Mean", "S.D.", "Min", "Max"])
    for group in ['Yearly general sentiment', 'Yearly herring sentiment', 'Yearly cod sentiment']:
        stats = calculate_summary_stats(data[group])["Sentiment Index"]
        ws.append([group] + list(stats.values))
    
    wb.save(file_path)
    
data = {
    'Quarterly general sentiment': sentiment_all_quarterly,
    'Quarterly herring sentiment': sentiment_herring_quarterly,
    'Quarterly cod sentiment': sentiment_cod_quarterly,
    'Yearly general sentiment': sentiment_all,
    'Yearly herring sentiment': sentiment_cod,
    'Yearly cod sentiment': sentiment_herring
}

# Save results
output_path = 'Results/sentiments_time_series_fig4a_4b.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    for sheet_name, df in data.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
create_summary_statistics_excel(data, 'Results/sentiments_summary_statistics_tabS5.xlsx')

###############################################################################
# Generate figure 3
###############################################################################

# Set color for quota and sentiment lines
col_quota = 'black' 
col_sent = 'black' 

fig, ax1 = plt.subplots(dpi = 300)

# Plot the rolling mean of the sentiment index
ax1.plot(sentiment_all_quarterly['Date'][4:-1], sentiment_all_quarterly['Sentiment Index'].rolling(window=3).mean()[4:-1], color='black', label='Sentiment Index')

# Set the x and y limits and formatting
ax1.set_xlim(datetime(2009, 1, 1), datetime(2021, 12, 31))
ax1.tick_params(axis='y', which='both', left=True, labelleft=True)
ax1.tick_params(axis='x', labelsize=18)
ax1.set_ylabel('Sentiment', fontsize=24)
ax1.set_xticks(ax1.get_xticks()[1:-1])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.yaxis.set_major_locator(AutoLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.yaxis.set_major_locator(MultipleLocator(0.05))
ax1.set_ylim(-0.4, -0.1)

# Updated dictionary for abbreviations
annotations = {
    'A': 'Cod stocks doubled',
    'B': 'Strong herring quota reduction',
    'C': 'High cod stocks',
    'D': 'Recovering fish stocks',
    'E': 'Herring quota increase',
    'F': 'Bad year for cod offspring',
    'G': 'Strong cod and herring quota reduction',
    'H': 'Fishing ban in cod spawning areas',
    'I': 'Strong cod quota and stock increase',
    'J': 'Baltic Sea Herring loses MSC certification',
    'K': 'Cod stocks almost at tipping point',
    'L': 'Strong cod quota and stock reduction'
}

# Updated list of important dates with new labels
important_dates = [
    (datetime(2009, 10, 21), 'A'),
    (datetime(2010, 7, 21), 'B'),
    (datetime(2011, 7, 27), 'C'),
    (datetime(2012, 7, 25), 'D'),
    (datetime(2013, 1, 23), 'D'),
    (datetime(2014, 10, 14), 'E'),
    (datetime(2015, 10, 23), 'F'),
    (datetime(2016, 7, 23), 'G'),
    (datetime(2017, 1, 11), 'H'),
    (datetime(2018, 1, 10), 'I'),
    (datetime(2018, 10, 15), 'J'),
    (datetime(2019, 7, 15), 'K'),
    (datetime(2019, 10, 15), 'L'),
]

for date, label in important_dates:
    date_index = sentiment_all_quarterly[4:-1][sentiment_all_quarterly[4:-1].iloc[:,0] == date].index
    important_value = sentiment_all_quarterly['Sentiment Index'].rolling(window=3).mean()[date_index]
    
    ax1.plot(date, important_value, 'bo', markersize=10)  
    ax1.annotate(label, xy=(date, important_value), xytext=(date, important_value - 0.02),
                 textcoords='data', fontsize=14, ha='center', va='top')

# Custom legend in multiple rows
handles = [plt.Line2D([0], [0], color='w', label=f'{abbr}: {full_text}')
           for abbr, full_text in annotations.items()]
ax1.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3, fontsize=10)

plt.savefig(r'Figures\figure3.eps', format='eps', bbox_inches='tight')
plt.show()

###############################################################################
# Generate figure 4
###############################################################################

fig, (ax1,ax3) = plt.subplots(1,2, figsize=(27.5,6), dpi = 300) 
 
ax2 = ax1.twinx() 
ax1.plot(quota_dates[1:], cod_quota.iloc[:,1], color = col_quota, linestyle = '--') 
ax2.plot(sentiment_cod_quarterly['Date'][1:-1], sentiment_cod_quarterly['Sentiment Index'].rolling(window=3).mean()[1:-1], color = col_sent)
ax1.hlines(y=0, color='black', linestyle='-', xmin = quota_dates.iloc[0], xmax = quota_dates.iloc[-1]) 
 
ax1.set_xlim(datetime(2008, 9, 1), datetime(2021, 12, 31))

ax1.tick_params(axis='y', colors=col_quota, labelsize = 18) 
ax2.tick_params(axis='y', colors=col_sent, labelsize = 18) 
 
ax1.tick_params(axis='x', labelsize = 18) 
 
ax1.set_ylabel('Cod quota (% change)', fontsize = 24) 
ax2.set_ylabel('Sentiment', fontsize = 24) 
 
ax1.set_title('(a) Cod', y=-0.32, fontsize = 24) 
 
ax1.set_xticks(ax1.get_xticks()[1:-1]) 
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45) 
 
ax1.yaxis.label.set_color(col_quota) 
ax2.yaxis.label.set_color(col_sent) 
 
ax1.xaxis.set_major_locator(mdates.YearLocator()) 
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 

ax1.set_ylim(-100, 80)
ax2.set_ylim(-0.5, -0.1)
 
###

ax4 = ax3.twinx()
ax3.plot(quota_dates[1:], herring_quota.iloc[:,1], color = col_quota, linestyle='--') 
ax4.plot(sentiment_herring_quarterly['Date'][1:-1], sentiment_herring_quarterly['Sentiment Index'].rolling(window=3).mean()[1:-1], color = col_sent) 
ax3.hlines(y=0, color='black', xmin = quota_dates.iloc[0], xmax = quota_dates.iloc[-1]) 

ax3.set_xlim(datetime(2008, 9, 1), datetime(2021, 12, 31))

ax3.tick_params(axis='y', colors=col_quota, labelsize = 18) 
ax4.tick_params(axis='y', colors=col_sent, labelsize = 18) 
 
ax3.tick_params(axis='x', labelsize = 18) 
 
ax4.set_ylabel('Sentiment', fontsize = 24) 
ax3.set_ylabel( 'Herring quota (% change)', fontsize = 24) 
 
ax3.set_title('(b) Herring', y=-0.32, fontsize = 24) 
 
ax3.set_xticks(ax3.get_xticks()[1:-1]) 
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45) 
 
ax3.yaxis.label.set_color(col_quota) 
ax4.yaxis.label.set_color(col_sent) 
 
ax3.xaxis.set_major_locator(mdates.YearLocator()) 
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 

ylim1_ax1 = ax1.get_ylim()
ylim1_ax2 = ax2.get_ylim()
ylim1_ax3 = ax3.get_ylim()
ylim1_ax4 = ax4.get_ylim()

ax3.set_ylim(-100, 80)
ax4.set_ylim(-0.5, -0.1)
 
plt.subplots_adjust(wspace=0.4, hspace=0) 
plt.savefig(r'Figures\figure4.svg', format='svg', bbox_inches='tight')
plt.show() 

###############################################################################

# Calculate Summary Statistics 
sentiment_all['Sentiment Index'].describe()[['mean', 'std', 'min', 'max']]
sentiment_herring['Sentiment Index'].describe()[['mean', 'std', 'min', 'max']]
sentiment_cod['Sentiment Index'].describe()[['mean', 'std', 'min', 'max']]

sentiment_all_quarterly['Sentiment Index'].describe()[['mean', 'std', 'min', 'max']]
sentiment_herring_quarterly['Sentiment Index'].describe()[['mean', 'std', 'min', 'max']]
sentiment_cod_quarterly['Sentiment Index'].describe()[['mean', 'std', 'min', 'max']]

sentiment_all['pos_sent_ratio'].describe()[['mean', 'std', 'min', 'max']]
sentiment_all['neu_sent_ratio'].describe()[['mean', 'std', 'min', 'max']]
sentiment_all['neg_sent_ratio'].describe()[['mean', 'std', 'min', 'max']]