#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:44:46 2023

@author: mfeene
"""


# -------------------------
#  2. Creating the Training Dataset
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# -------------------------
# Data Loading 
# -------------------------
### Convert the game data file into a file for modeling
## First find the differences in value between each of the numerical variables from winning team to losing team
games_with_season_stats = pd.read_csv('/Users/mfeene/Desktop/MarchMadness_2023/2023_additional_variables/games_with_season_stats_addl.csv')
del games_with_season_stats['Unnamed: 0']


# -------------------------
# Data Manipulation 
# -------------------------
model_data = games_with_season_stats[['Season', 'WTeamID', 'LTeamID']]
model_data = model_data.rename(columns = {'WTeamID': 'Team1', 'LTeamID': 'Team2'})

model_data['WLoc'] = games_with_season_stats['WLoc']
model_data['WinsDiff'] = games_with_season_stats['WTeam_Wins'] - games_with_season_stats['LTeam_Wins']
model_data['LossesDiff'] = games_with_season_stats['WTeam_Losses'] - games_with_season_stats['LTeam_Losses']
model_data['AvgRankDiff'] = games_with_season_stats['WTeam_AvgRank'] - games_with_season_stats['LTeam_AvgRank']
model_data['BestRankDiff'] = games_with_season_stats['WTeam_BestRank'] - games_with_season_stats['LTeam_BestRank']
model_data['WorstRankDiff'] = games_with_season_stats['WTeam_WorstRank'] - games_with_season_stats['LTeam_WorstRank']
model_data['WinPctDiff'] = games_with_season_stats['WTeam_WinPct'] - games_with_season_stats['LTeam_WinPct']
model_data['FGRatioDiff'] = games_with_season_stats['WTeam_FG_ratio'] - games_with_season_stats['LTeam_FG_ratio']
model_data['3PTDiff'] = games_with_season_stats['WTeam_3PT_ratio'] - games_with_season_stats['LTeam_3PT_ratio']
model_data['ATORatioDiff'] = games_with_season_stats['WTeam_ATO_ratio'] - games_with_season_stats['LTeam_ATO_ratio']
model_data['Outcome'] = 1


# -------------------------
# Data Manipulation 2
# -------------------------
# Randomly sample 50% of the dataframe to simulate the other team winning
# Multiply the diff variables by -1
model_data_to_swap = model_data.sample(frac = 0.5, random_state = 123)

model_data_to_swap['WinsDiff'] = model_data_to_swap['WinsDiff']*-1
model_data_to_swap['LossesDiff'] = model_data_to_swap['LossesDiff']*-1
model_data_to_swap['AvgRankDiff'] = model_data_to_swap['AvgRankDiff']*-1
model_data_to_swap['BestRankDiff'] = model_data_to_swap['BestRankDiff']*-1
model_data_to_swap['WorstRankDiff'] = model_data_to_swap['WorstRankDiff']*-1
model_data_to_swap['WinPctDiff'] = model_data_to_swap['WinPctDiff']*-1
model_data_to_swap['FGRatioDiff'] = model_data_to_swap['FGRatioDiff']*-1
model_data_to_swap['3PTDiff'] = model_data_to_swap['3PTDiff']*-1
model_data_to_swap['ATORatioDiff'] = model_data_to_swap['ATORatioDiff']*-1

# Swap the positions of WTeamID and LTeamID
model_data_to_swap = model_data_to_swap[['Season', 'Team2', 'Team1', 'WLoc', 'WinsDiff', 'LossesDiff', 'AvgRankDiff', 
                                         'BestRankDiff', 'WorstRankDiff',
                                         'WinPctDiff', 'FGRatioDiff', '3PTDiff', 'ATORatioDiff', 
                                         'Outcome']]

# These will have their outcome variables switched to 0
model_data_to_swap['Outcome'] = model_data_to_swap['Outcome'].replace([1], [0])

# Rest of the dataframe, leave as is
model_data_orig = model_data.loc[~model_data.index.isin(model_data_to_swap.index)]


# -------------------------
# Final Modeling Dataset
# -------------------------
## Put the two dataframes back together
final_model_data = pd.concat([model_data_orig, model_data_to_swap], axis = 0)
final_model_data.to_csv('final_model_data_addl.csv')


