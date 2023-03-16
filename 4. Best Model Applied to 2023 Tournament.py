#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:44:46 2023

@author: mfeene
# Run in Google colab for GPU
"""


# -------------------------
#  4. Applying the Model to the 2023 Tournament
# -------------------------

# Run first for reproducability
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

# #https://www.kaggle.com/competitions/mens-march-mania-2022/discussion/308685
# !pip install binarytree==6.2.0
# !pip install bracketeer==0.2.0
# !pip install setuptools_scm==6.0.1

import pandas as pd
import pickle
from bracketeer import build_bracket

# -------------------------
#  Data Processing
# -------------------------
# Read in data
submissions = pd.read_csv('/content/SampleSubmission2023.csv')
data = pd.read_csv('/content/team_stats_ratios_2023_addl.csv')
del data['Unnamed: 0']

# Model trained on historical data from 2003, apply to the 2023 matchups
# split game identifier by specified delimiter
matchups = pd.DataFrame()
matchups[['Season','Team1ID', 'Team2ID']] = submissions.ID.str.split("_", expand = True)
matchups['Season'] = pd.to_numeric(matchups['Season'])
matchups['Team1ID'] = pd.to_numeric(matchups['Team1ID'])
matchups['Team2ID'] = pd.to_numeric(matchups['Team2ID'])

# Bring in the team stats for 2023 
team1_metrics_added = pd.merge(matchups, data, how = 'left', left_on = ['Season', 'Team1ID'], right_on = ['Season', 'TeamID'])
team1_metrics_added.rename(columns = {'Wins': 'Team1_Wins',
                                           'Losses': 'Team1_Losses',
                                           'AvgRank': 'Team1_AvgRank',
                                           'BestRank': 'Team1_BestRank',
                                           'WorstRank': 'Team1_WorstRank',
                                           'WinPct': 'Team1_WinPct',
                                           'FG_ratio' : 'Team1_FG_ratio',
                                           '3PT_ratio': 'Team1_3PT_ratio',
                                           'ATO_ratio': 'Team1_ATO_ratio'}, inplace = True)
 
team2_metrics_added = pd.merge(team1_metrics_added, data, how = 'left', left_on = ['Season', 'Team2ID'], right_on = ['Season', 'TeamID'])
team2_metrics_added.rename(columns = {'Wins': 'Team2_Wins',
                                           'Losses': 'Team2_Losses',
                                           'AvgRank': 'Team2_AvgRank',
                                           'BestRank': 'Team2_BestRank',
                                           'WorstRank': 'Team2_WorstRank',
                                           'WinPct': 'Team2_WinPct',
                                           'FG_ratio' : 'Team2_FG_ratio',
                                           '3PT_ratio': 'Team2_3PT_ratio',
                                           'ATO_ratio': 'Team2_ATO_ratio'}, inplace = True)
# Add this in
team2_metrics_added['WLoc'] = 0
 
# Get data into this format for the 2023
model_2023_data = pd.DataFrame()
model_2023_data = team2_metrics_added[['Season', 'Team1ID', 'Team2ID']]
 
model_2023_data['WLoc'] = team2_metrics_added['WLoc']
model_2023_data['WinsDiff'] = team2_metrics_added['Team1_Wins'] - team2_metrics_added['Team2_Wins']
model_2023_data['LossesDiff'] = team2_metrics_added['Team1_Losses'] - team2_metrics_added['Team2_Losses']
model_2023_data['AvgRankDiff'] = team2_metrics_added['Team1_AvgRank'] - team2_metrics_added['Team2_AvgRank']
model_2023_data['BestRankDiff'] = team2_metrics_added['Team1_BestRank'] - team2_metrics_added['Team2_BestRank']
model_2023_data['WorstRankDiff'] = team2_metrics_added['Team1_WorstRank'] - team2_metrics_added['Team2_WorstRank']
 
model_2023_data['WinPctDiff'] = team2_metrics_added['Team1_WinPct'] - team2_metrics_added['Team2_WinPct'] 
model_2023_data['FGRatioDiff'] = team2_metrics_added['Team1_FG_ratio'] - team2_metrics_added['Team2_FG_ratio']
model_2023_data['3PTDiff'] = team2_metrics_added['Team1_3PT_ratio'] - team2_metrics_added['Team2_3PT_ratio']
model_2023_data['ATORatioDiff'] = team2_metrics_added['Team1_ATO_ratio'] - team2_metrics_added['Team2_ATO_ratio']
 
 
# Load trained model
with open('mm_hypermodel.pkl', 'rb') as fid:
    mm_hypermodel = pickle.load(fid)
       
# Load scaler
scalerfile = 'scaler.save'
sc = pickle.load(open(scalerfile, 'rb'))
 

# -------------------------
# Generating Predictions
# -------------------------
# Read in data
X = model_2023_data.iloc[:, 3:].values
 
# Apply scaler to data
X = sc.transform(X)

# Predict probabilities of team 1 winning each matchup
predictions = mm_hypermodel.predict(X) #predict proba for non neural nets
predictions = predictions.flatten()


# -------------------------
# 2023 Predictions
# -------------------------
# Read in sample predictions
template = pd.read_csv('/content/SampleSubmission2023.csv')
spreadsheet = template.iloc[:, :].values

# Add the prediction to each prediction index
for i in range(0, len(predictions)):
     spreadsheet[i][1] = round(predictions[i], 5)
 
# Create dataframe to match sample submissions spreadsheet
results = pd.DataFrame(data = spreadsheet, columns=['ID', 'Pred'])

# Save new submissions spreadsheet as csv 
results.to_csv('Submission2023_kerastuner_march15.csv', sep = ',', encoding = 'utf-8', index = False)


# -------------------------
# 2023 Bracket
# -------------------------
#from IPython.display import display, Image

b = build_bracket(
    outputPath='MNCAA2023_kerastuner_march15.png',
    teamsPath='/content/MTeams.csv',
    seedsPath='/content/MNCAATourneySeeds.csv',
    submissionPath='/content/Submission2023_kerastuner_march15.csv',
    slotsPath='/content/MNCAATourneySlots.csv',
    year=2023
)

#display(Image(filename='/content/MNCAA2023_kerastuner_march15.png'))

#model = ...  # Get model (Sequential, Functional Model, or Model subclass)
#mm_hypermodel.save('/content/')

#with open('mm_hypermodel.pkl', 'wb') as fid:
#    pickle.dump(mm_hypermodel, fid)

