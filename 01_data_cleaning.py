# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 19:42:46 2025

@author: saika
"""

import pandas as pd
import numpy as np

daily_tracks = pd.read_csv('Apple Music - Play History Daily Tracks.csv')
data2 = pd.read_csv('Apple Music - Top Content.csv')
Library_tracks = pd.read_json('Apple Music Library Tracks.json')

data = pd.merge(daily_tracks, Library_tracks,
              left_on='Track Identifier',
              right_on = 'Apple Music Track Identifier',
              how ='inner')

print(data.shape)


print(data['End Reason Type'].value_counts())

df = data[['Media type','Date Played','Hours','Play Duration Milliseconds','End Reason Type',
           'Play Count','Skip Count_x','Content Type','Title','Sort Name','Artist','Album',
           'Album Artist','Genre','Grouping','Track Year','Track Play Count','Rating','Album Rating','Skip Count_y'
           ]]


df = df.drop(columns = ['Play Count','Skip Count_x'])

#df.to_csv('data_handpick.csv',index=False)

play_count = df[['Track Play Count','Skip Count_y']]

#%%

print(df['Track Play Count'].dtype)
print(df['Skip Count_y'].dtype)

#%%
print(df['Grouping'].value_counts())
#%%
print(df['Rating'].value_counts())

#%%
print(df['Album Rating'].value_counts())

#%%
print(df['Album Artist'].value_counts())
print(df['Album Artist'].isnull().sum())

#%%
df = df.drop(columns =['Grouping','Rating','Album Rating','Sort Name','Album Artist'])
        
isnull_drop1 =  df.isnull().sum()

#%%

isnull = df.isnull().sum()

#%%
print(df['Media type'].value_counts())

#%%
media_null = df[df['Media type'].isnull()]

#%%

df['Media type'] = df['Media type'].fillna('AUDIO')

isnull = df.isnull().sum()

#%%

df = df.dropna()

isnull = df.isnull().sum()

print(df['Content Type'].value_counts())

#%%
# filtering duplicates in title #
df_filtered = df.drop_duplicates(subset=['Title']).copy()
df_filtered.to_csv('df_filtered.csv',index=False)

isnull = df_filtered.isnull().sum()
print("\nChecking Nulls in the dataset\n")
print(df_filtered.isnull().sum())
#%%
# milliseconds to seconds

df_filtered['Play Duration Seconds'] = df_filtered['Play Duration Milliseconds']/1000

# condition check for 0 second length play durations
condition = df_filtered['Play Duration Seconds'] < 60

#replacing 0 and <60 with mean() of the col duration 
df_filtered.loc[condition, 'Play Duration Seconds'] = df_filtered['Play Duration Seconds'].mean()
df_filtered_1 = df_filtered.drop(columns=['Play Duration Milliseconds'])

#%% 

# filtering 0 in track year , songs need to have a valid year release date
zeros_count = (df_filtered_1['Track Year'] == 0).sum()
print('Zeros in Track Year', zeros_count)
#%% 

# replacing 0 in track year with 1995
df_filtered_1.loc[df_filtered['Track Year'] == 0, 'Track Year'] = 1995

zeros_count = (df_filtered_1['Track Year'] == 0).sum()
print('Zeros in Track Year', zeros_count)

#%%

df_filtered_1.to_csv('df_clean_V1.1.csv',index=False)
print('\n\n********** Data cleaning ends here **********\n\n\n')

