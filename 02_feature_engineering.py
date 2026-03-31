# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 00:38:55 2025

@author: saika
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
print("\n\n**********Feature Engineering**********\n\n")

# Re-loading clean dataset from the previous py
df = pd.read_csv('df_clean_V1.1.csv')
# convert Hours column to a list of integers
df['Hours_list'] = df['Hours'].apply(
    lambda x: list(map(int, str(x).split(',')))
)
#%%

def encode_periods(hours):
    periods = {
        'Morning': 0,
        'Afternoon': 0,
        'Night': 0
    }

    for h in hours:
        if 5 <= h <= 11:
            periods['Morning'] = 1
        elif 12 <= h <= 16:
            periods['Afternoon'] = 1
        else:  # night = 17–23 OR 0–4
            periods['Night'] = 1

    return pd.Series(periods)

# Apply multi-hot encoding
df[['Morning', 'Afternoon', 'Night']] = df['Hours_list'].apply(encode_periods)

#%%

"""
We have Track play count and skip_count y, idea is to calculate the ratio
between play counts and skip counts and based on that ratio, we will create
"Like/Dislike" column.
  
Says:
play count = 10, skip count =4 --> skip & play ration is 3/10 which is 0.4
imagine a song plays 10 times and 3 skips considers like.

play count = 5, skip count =4 --> skip & play ration is 4/5 which is 0.8
imagine a song play 5 times but 4 skips consider dislike.

we apply this method to evaluate the Like/Dislike based on skip/play ratio.

"""
#%%
# create a function that takes row from df
def cal_ratio(row):
    #assign values from each column
    play = row['Track Play Count']
    skip = row['Skip Count_y']
    
    #condition check --> only songs that play once will calculate
    if play >0:
        return skip/play
        # return the ratio if plays
    elif play ==0 and skip >0:
        # return total dislike if no play and skips 
        return 1.0
    elif play ==0 and skip == 0:
        #no play, no skip, natural or never listen before
        return np.nan
    else:
        return np.nan
    
def Like_dislike (ratio):
    
    if pd.isna(ratio):
        return "Neutral"
    elif ratio < 0.5:
        return "Like"
    else:
        return "Dislike"

# function call to calculate ratio and set a column within df
df['Skip Ratio'] = df.apply(cal_ratio, axis=1)
# function call to set like/Neutral/dislike and set a column within df
df['Like/Dislike'] = df['Skip Ratio'].apply(Like_dislike)
#%%
counts = df['Like/Dislike'].value_counts()
        
        # Create the Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Distribution of Like/Dislike/Neutral')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()       
# Save the plot
plt.savefig('like_dislike_pie_chart.png')
plt.close() 
        
print("Pie chart created successfully: 'like_dislike_pie_chart.png'")
print(f"Like/Dislike column counts:\n\n{counts}")
print("ratio: 0.1")

#%%

#%%

df_SortPlayDuration = df.sort_values('Play Duration Seconds', ascending=False)
print(df_SortPlayDuration[['Play Duration Seconds', 'Title', 'Artist']].head(5))

print(df['End Reason Type'].value_counts())

df_engineering_1 = df.drop(columns = ['Media type','Hours','Content Type',
                                      'Track Play Count','Skip Count_y','Hours_list','Skip Ratio'])

#%%
# Date Played into Date_time format
df_engineering_1['Date Played'] = pd.to_datetime(df_engineering_1['Date Played'], format='%Y%m%d')

print(df_engineering_1['Date Played'])

#%%
df_engineering_1['Year played'] = df_engineering_1['Date Played'].dt.year
df_engineering_1['Month played'] = df_engineering_1['Date Played'].dt.month
df_engineering_1['Day Played'] = df_engineering_1['Date Played'].dt.day

df_engineering_2 = df_engineering_1.drop(columns = ['Date Played'])

#%%

print(df_engineering_2.dtypes)

#%%

# Cleaning Genre Feature

# Challenges:
# 1. Music categorization can  be subjective
# 2. The genre names here are not consistent ex. variations of Hip-hop/rap and hip-hop & rap 

#df_filtered['Genre'].value_counts().head(25).plot(kind='bar')
#print(df_filtered['Genre'].value_counts().head(10))
print(f'Number of unique Genres before Cleaning: {df_engineering_2['Genre'].nunique()}')

# Normalize string
df_engineering_2['Genre_clean'] = (
    df_engineering_2['Genre']
    .str.strip()        # remove leading/trailing spaces
    .str.lower()        # make lowercase
    .str.replace(r'\s+', ' ', regex=True)  # collapse multiple spaces
)

# Create mapping for known variants
# note: ID3v1 genres 131 is indie and 147 is synthpop
mapping = {
    'hip-hop/rap': 'hip-hop & rap',
    'hip-hop': 'hip-hop & rap',
    'rap & hip hop': 'hip-hop & rap',
    'rap & hip-hop': 'hip-hop & rap',
    'rap': 'hip-hop & rap',
    'hip hop/rap': 'hip-hop & rap',
    'hip hop / rap': 'hip-hop & rap',
    'hip hop': 'hip-hop & rap',
    'indie': 'indie rock',
    'rock/indie': 'indie rock',
    'indie/rock': 'indie rock',
    'indie pop/rock': 'indie rock',
    'alternative/indie': 'indie rock',
    'world indie': 'indie rock',
    '131': 'indie rock',
    'general indie rock': 'indie rock',
    'unknown': 'indie rock',
    'mixed genre': 'indie rock',
    'cosmic tones for mental therapy': 'indie rock',
    'other': 'indie rock',
    'ambient': 'indie rock',
    'indie Pop': 'pop',
    'indie pop': 'pop',
    'punk': 'punk rock',
    'alternative & punk': 'punk rock',
    'general alternative': 'alternative',
    'alternative rock': 'alternative',
    'alternrock': 'alternative',
    'r&b': 'r&b/soul',
    'soul and r&b': 'r&b/soul',
    'indie electronic': 'electronic',
    'electronica/dance': 'electronic',
    'electronica': 'electronic',
    'alternativt': 'electronic',
    'traditional folk': 'folk',
    'indie/ folk': 'folk',
    'contemporary folk': 'folk',
    'alternative folk': 'folk',
    'alt-folk': 'folk',
    'indie folk': 'folk',
    'folk-rock': 'folk',
    'hard rock': 'rock',
    'rock & roll': 'rock',
    'roots rock': 'rock',
    'rock/pop': 'rock',
    'pop/rock': 'rock',
    'classic rock': 'rock',
    'genre': 'rock',
    'experimental rock': 'rock',
    'country & folk': 'country',
    'trash metal': 'metal',
    'death metal/black metal': 'metal' 
}

# new column for clean Genre
df_engineering_2['Genre_clean'] = df_engineering_2['Genre_clean'].replace(mapping)

# create list of main genres
main_genres = ['alternative', 'rock', 'indie rock', 'hip-hop & rap', 'punk rock',
               'pop', 'folk', 'country', 'singer/songwriter', 'electronic',
               'r&b/soul', 'jazz', 'children\'s music', 'spoken & audio',
               'soundtrack', 'reggae', 'metal', 'new wave', 'blues',
               'synthpop', 'funk', 'worldwide', 'post rock', 'americana']

# if not a main genre, group into 'other'
df_engineering_2["Genre_clean"] = df_engineering_2["Genre_clean"].apply(
    lambda g: g if g in main_genres else "other"
)

print(f'Number of unique Genres after Cleaning: {df_engineering_2['Genre_clean'].nunique()}')

df_engineering_2['Genre_clean'].value_counts().head(10).plot(kind='bar')
plt.show()
# Save the plot
plt.savefig('Genre.png')
plt.close()


df_engineering_3 = df_engineering_2.drop(columns='Genre')
#%%

# Title and Album doesn't matter in this stage, but Artist matters since people are the fan of artists
df_engineering_4 = df_engineering_3.drop(columns=['Title','Album'])


#%%


like_mapper = {
    'Like': 1,
    'Dislike': 0,
    'Neutral': 0
}

df_engineering_4['Is_like'] = df_engineering_4['Like/Dislike'].replace(like_mapper).astype('int64')

df_engineering_5 = df_engineering_4.drop(columns = 'Like/Dislike')

#%%
df_engineering_5 = df_engineering_5.rename(columns={'Genre_clean':'Genre'})


#%%

df_engineering_5.to_csv("df_engineering_5.csv",index=False)

#%%
scaler_standard = StandardScaler()
df_engineering_6 = df_engineering_5.copy()

cols_to_normalize = ['Track Year', 'Play Duration Seconds','Morning','Afternoon','Night', 'Year played', 'Month played', 'Day Played']

df_engineering_6[cols_to_normalize] = scaler_standard.fit_transform(df_engineering_6[cols_to_normalize])
#%%

# df_engineering_7 =df_engineering_6.drop(columns=['Morning','Afternoon','Night'])
df_engineering_6.to_csv('df_engineering_6.csv')
#%%
# getting dummies

df_dummies = pd.get_dummies(df_engineering_6,drop_first=True,dtype=int)

#%%
# corr_matrix = df_dummies.corr()
    
# print("Correlation Matrix:")
# print(corr_matrix)
    
#     # Plot the heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()

#%%
df_dummies.to_csv("df_dummies.csv",index=False)
print('\n\n********** Feature engineering ends here **********\n\n\n')