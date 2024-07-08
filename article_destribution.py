# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

from nltk.tokenize import word_tokenize 

# Load Data
data = pd.read_csv(r'Data\fishery_lemmas_articles.csv')

# Ensure 'Date' column is in datetime format
data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')

# Load the quota decision dates
dates = pd.read_excel('Data\data, fish & fisheries, SD22-24.xlsx', sheet_name='dates, advice - quota ')['quota decision']
dates = dates.dropna()
dates = pd.to_datetime(dates)
dates = dates[dates >= "2009-01-01"]

# Count number of articles per year
year_count = data.groupby(pd.Grouper(key='Date', freq='Y')).Date.count()

# Calculate the monthly average share of articles
month_count = data.groupby(pd.Grouper(key='Date', freq='M')).Date.count()
month_count.index = pd.to_datetime(month_count.index)
monthly_averages = month_count.groupby(month_count.index.month).mean() / 100

# Create a DataFrame for plotting
month_count_prop_all = pd.DataFrame({'proportions': [0] * 12})
month_count_prop_all['proportions'][0:12] = monthly_averages

# Generate figure 1
year_count = data.groupby(pd.Grouper(key='Date', freq='Y')).Date.count()

plt.figure(dpi=300, figsize=(8, 4))
plt.bar(year_count.index.year, year_count, width = 0.75, color='grey') #'k-'
plt.xticks(year_count.index.year, fontsize = 7)
plt.ylabel('Number of published articles')
plt.savefig(r'Figures\figure1.svg', format='svg', bbox_inches='tight')
plt.savefig(r'Figures\figure1.eps', format='eps', bbox_inches='tight')
plt.show()

# Save results
yearly_count = pd.DataFrame({'Year': year_count.index.year, 'Count': year_count.values})
output_path = os.path.join('Results', 'yearly_article_count_fig1.xlsx')
yearly_count.to_excel(output_path, index=False)

# Generate figure 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi = 300)  

# Generate figure 2(a)
ax1.bar(range(1, 13), month_count_prop_all['proportions'], color='0.4')
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax1.set_ylabel("Average yearly share of articles per calendar month", fontsize=14)
ax1.set_title("(a) Average monthly share of articles", y=-0.17, fontsize=16)

# Function to count articles per day around specific dates
def count_article_per_day(dates,data,n):
    """
    Count the number of articles per day around specific dates.
    
    Args:
        dates (pd.Series): Dates around which to count articles.
        data (pd.DataFrame): DataFrame containing articles with 'Date' column.
        n (int): Number of days before and after the dates to consider.
    
    Returns:
        list: List of article counts per day around each date.
    """
    yearly_totals = data.groupby(data['Date'].dt.year).size()
    all_counts = []
    
    for date in dates:
        year = date.year
        year_total = yearly_totals.get(year, 1) 
        
        mask = (data['Date'] >= date - pd.Timedelta(days=n)) & (data['Date'] <= date + pd.Timedelta(days=n))
        filtered_df = data[mask]
        counts_df = filtered_df.groupby('Date').size()
        reindexed_df = counts_df.reindex(pd.date_range(start=date - pd.Timedelta(days=n), end=date + pd.Timedelta(days=n))).fillna(0)
        fractional_counts = reindexed_df.divide(year_total)
        
        all_counts.append(fractional_counts.values)
        
    return(all_counts)

# Select number of days before and after specified dates
n = 4
all_counts = count_article_per_day(dates,data,n)
all_counts_array = np.array(all_counts)

# Calculate the mean and standard deviation across all counts for each day
means = np.mean(all_counts_array, axis=0)
std_devs = np.std(all_counts_array, axis=0)

# Calculate the standard errors
standard_errors = std_devs / np.sqrt(len(dates))

# Compute the 95% confidence intervals
confidence_intervals = standard_errors * 1.96

# Generate figure 2(b)
days = np.arange(-n, n + 1)
ax2.bar(days, means, yerr=confidence_intervals, capsize=5, color='grey', edgecolor='black')
ax2.set_xlabel('Days relative to date of quota announcement', fontsize=14)
ax2.set_ylabel('Average yearly share of articles per day', fontsize=14)
ax2.set_title("(b) Average daily share of articles around quota announcements", y=-0.17, fontsize=14)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xticks(days)  # Set ticks for every day in the range

plt.tight_layout()
plt.savefig(r'Figures\figure2.svg', format='svg', bbox_inches='tight')
plt.show()

# Save results
monthly_means = pd.DataFrame({'Months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 'Monthly shares': month_count_prop_all['proportions']})
daily_means_quota = pd.DataFrame({'Days': days, 'Daily shares': means})

output_path = os.path.join('Results', 'monthly_article_means_fig2a.xlsx')
monthly_means.to_excel(output_path, index=False)

output_path = os.path.join('Results', 'daily_article_means_quota_fig2b.xlsx')
daily_means_quota.to_excel(output_path, index=False)

###############################################################################

# Generate figure S2

# Load dates for quota advices
dates = pd.read_excel('Data\data, fish & fisheries, SD22-24.xlsx', sheet_name = 'dates, advice - quota ')['advice']
dates = dates[:-1] 
dates = [date for date in dates if pd.notna(date) and date != "None\xa0"]

all_counts = count_article_per_day(dates,data,n)
all_counts_array = np.array(all_counts)

# Calculate the mean and standard deviation across all counts for each day
means = np.mean(all_counts_array, axis=0)
std_devs = np.std(all_counts_array, axis=0)

# Calculate the standard errors
standard_errors = std_devs / np.sqrt(len(dates))

# Compute the 95% confidence intervals
confidence_intervals = standard_errors * 1.96

# Average yearly share of articles per day around quota advice dates
fig, ax = plt.subplots(figsize=(10, 6), dpi = 300)
ax.bar(days, means, yerr=confidence_intervals, capsize=5, color='grey', edgecolor='black')
ax.set_xlabel('Days relative to date of quota advice', fontsize=14)
ax.set_ylabel('Average yearly share of articles per day', fontsize=14)
ax.set_title("Average daily share of articles around quota advice", y=-0.17, fontsize=14)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xticks(days)  # Set ticks for every day in the range

plt.tight_layout()  
plt.savefig(r'Figures\figureS2.svg', format='svg', bbox_inches='tight')
plt.savefig(r'Figures\figureS2.eps', format='svg', bbox_inches='tight')
plt.show()

# Save results
figs2 = pd.DataFrame({'Days': days, 'Daily shares': means})

daily_means_advice = pd.DataFrame({'Days': days, 'Daily shares': means})

output_path = os.path.join('Results', 'daily_article_means_advice_figS2.xlsx')
daily_means_quota.to_excel(output_path, index=False)

###############################################################################

# Generate figure S3
dates = pd.read_excel('Data\data, fish & fisheries, SD22-24.xlsx', sheet_name = 'dates, advice - quota ')['quota decision']
dates = dates[:-1]
dates = dates[dates >= "2009-01-01"]
dates = dates.reset_index(drop = True)

# Initialize lists to hold the share of articles for each species
hering_share = []
cod_share = []
salmon_share= []
sprat_share = []
plaice_share = []

stakeholder = pd.DataFrame()

data['tokens'] = [word_tokenize(lemmas) for lemmas in data['Lemmas']]

years = sorted(list(set(data['year']))) 

for idx,_ in enumerate(years[1:]):

    start_date = dates[idx]
    end_date = dates[idx+1]
          
    # reorder yearly data based "fishery year" 
    yearly_data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
    
    #yearly_data = reorder_data(data_art, year, start_month , end_month)
    yearly_lemmas = ' '.join(yearly_data['Lemmas'])
       
    hering_share.append(len([1 for article in yearly_data['tokens'] if ('hering' or 'heringe') in article])/len(yearly_data))
    cod_share.append(len([1 for article in yearly_data['tokens'] if ('dorsch' or 'dorsche' or 'kabeljaue' or 'kabeljau') in article])/len(yearly_data))
    sprat_share.append(len([1 for article in yearly_data['tokens'] if 'sprotte' in article])/len(yearly_data))
    salmon_share.append(len([1 for article in yearly_data['tokens'] if ('lachs' or 'lachse') in article])/len(yearly_data))
    plaice_share.append(len([1 for article in yearly_data['tokens'] if 'scholle' in article])/len(yearly_data))

share = pd.DataFrame({'years': years[:-1], 'hering': hering_share, 'cod': cod_share, 
              'salmon': salmon_share, 'sprat': sprat_share, 'plaice': plaice_share})

fig, ax1 = plt.subplots(figsize=(10, 6), dpi = 300)

ax1.plot(share['years'], share['hering'], color='black', linestyle='-', label='Hering')
ax1.plot(share['years'], share['cod'], color='black', linestyle='--', label='Cod')
ax1.plot(share['years'], share['salmon'], color='black', linestyle='dashdot', label='Salmon')
ax1.plot(share['years'], share['sprat'], color='black', linestyle='dotted', label='Sprat')
ax1.plot(share['years'], share['plaice'], color='black', linestyle=(0, (1, 10)), label='Plaice')

ax1.set_ylabel('Share of articles', fontsize=14)
ax1.tick_params(axis='y', labelsize=12)

ax1.set_xticks(share['years'])
ax1.set_xticklabels(share['years'], rotation=45)

ax1.tick_params(axis='x', labelsize=12)

ax1.legend(frameon=False, prop={'size': 14}, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=5)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.minorticks_off()
plt.tight_layout()
plt.savefig(r'Figures\figureS3.svg', format='svg', bbox_inches='tight')
plt.savefig(r'Figures\figureS3.eps', format='svg', bbox_inches='tight')
plt.show()

# Calculate and print mean shares of articles for each species
print("Mean shares of articles:")
print("Hering:", sum(share['hering']) / len(share['hering']))
print("Cod:", sum(share['cod']) / len(share['cod']))
print("Plaice:", sum(share['plaice']) / len(share['plaice']))
print("Sprat:", sum(share['sprat']) / len(share['sprat']))
print("Salmon:", sum(share['salmon']) / len(share['salmon']))

# Save results
fish_shares = pd.DataFrame(share)

output_path = os.path.join('Results', 'fish_shares_figS3.xlsx')
fish_shares.to_excel(output_path, index=False)
