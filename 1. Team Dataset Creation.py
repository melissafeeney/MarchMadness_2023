#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:44:46 2023

@author: mfeene
"""


# -------------------------
#  1. Creating the Team Dataset
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Data Loading
# -------------------------
# Read in data
path = '/Users/mfeene/Desktop/MarchMadness_2023/2023_additional_variables/data/'

## Read in Data
massey_ord = pd.read_csv(path + 'MMasseyOrdinals_thru_Season2023_Day128.csv')
regseason = pd.read_csv(path + 'MRegularSeasonDetailedResults.csv')
postseason = pd.read_csv(path + 'MNCAATourneyDetailedResults.csv')

# Only consider 2003 and on
regseason = regseason[(regseason['Season'] >= 2003)]
postseason = postseason[(postseason['Season'] >= 2003)]
frames = [regseason, postseason]
games = pd.concat(frames)


# -------------------------
#  Team Descriptive Stats- Avg, Best and Worst Pom Rankings per Season per Team
# -------------------------
# Consider Pomeroy Rankings- note that this only goes through the regseason
pomeroy_rank = massey_ord[(massey_ord['SystemName'] == 'POM')]

# For each team in each year, record the avg ranking, best rank, worst rank
avg_rank = pomeroy_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).mean().reset_index()
best_rank = pomeroy_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).min().reset_index()
worst_rank = pomeroy_rank[['Season', 'TeamID', 'OrdinalRank']].groupby(['Season','TeamID']).max().reset_index()

avg_rank.columns = ['Season', 'TeamID', 'AvgRank']
best_rank.columns = ['Season', 'TeamID', 'BestRank']
worst_rank.columns = ['Season', 'TeamID', 'WorstRank']

pre_team_ranks = pd.merge(avg_rank, best_rank, how = 'left', left_on =['Season', 'TeamID'], right_on =['Season', 'TeamID'])
team_ranks = pd.merge(pre_team_ranks, worst_rank, how = 'left', left_on =['Season', 'TeamID'], right_on =['Season', 'TeamID']) ## contains each team's avg, best and worst rank er season


# -------------------------
#  Team Descriptive Stats- Wins and Losses, Win percentage, per Season per Team
# -------------------------
## Wins per season per team
wins_per_season = games[['Season', 'WTeamID']].groupby(['Season', 'WTeamID']).size().reset_index()
wins_per_season.columns = ['Season', 'TeamID', 'Wins']

## Losses per season per team
losses_per_season = games[['Season', 'LTeamID']].groupby(['Season', 'LTeamID']).size().reset_index()
losses_per_season.columns = ['Season', 'TeamID', 'Losses']

# Put all together
wins_losses = pd.merge(wins_per_season, losses_per_season, how = 'outer', on =['Season', 'TeamID'])
wins_losses  = wins_losses.sort_values(by = ['Season', 'TeamID']).reset_index(drop = True).fillna(0.0)
team_stats = pd.merge(wins_losses, team_ranks, how = 'outer', on =['Season', 'TeamID'])
team_stats = team_stats.sort_values(by = ['Season', 'TeamID']).reset_index(drop = True).fillna(0.0)

# Calculate wins percentage for each team each season
team_stats['WinPct'] = team_stats['Wins'] / (team_stats['Wins'] + team_stats['Losses'])


# -------------------------
#  More Team Descriptive Stats per Season per Team- FG ratio, 3PT ratio, ATO ratio
# -------------------------
# instances where the team won
winner_metrics = games[['Season', 'WTeamID', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WAst', 'WTO' ]].groupby(['Season', 'WTeamID']).sum().reset_index()
winner_metrics.rename(columns = {'WTeamID': 'TeamID'}, inplace = True)

# instances where the team lost
loser_metrics = games[['Season', 'LTeamID', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LAst', 'LTO']].groupby(['Season', 'LTeamID']).sum().reset_index()
loser_metrics.rename(columns = {'LTeamID': 'TeamID'}, inplace = True)

# Join to get total metrics per season per team
season_ratios = pd.merge(winner_metrics, loser_metrics, how = 'outer', on =['Season', 'TeamID'])
season_ratios = season_ratios.sort_values(by = ['Season', 'TeamID']).reset_index(drop = True).fillna(0.0)
season_ratios['FG_ratio'] = (season_ratios['WFGM'] + season_ratios['LFGM'])/(season_ratios['WFGA'] + season_ratios['LFGA'])
season_ratios['3PT_ratio'] = (season_ratios['WFGM3'] + season_ratios['LFGM3'])/(season_ratios['WFGA3'] + season_ratios['LFGA3'])
season_ratios['ATO_ratio'] = (season_ratios['WAst'] + season_ratios['LAst'])/(season_ratios['WTO'] + season_ratios['LTO'])
season_ratios = season_ratios[['Season', 'TeamID', 'FG_ratio', '3PT_ratio', 'ATO_ratio']]

# Add these ratios to the team_stats file
team_stats_ratios = pd.merge(team_stats, season_ratios, how = 'outer', on =['Season', 'TeamID'] )
team_stats_ratios.to_csv('team_stats_ratios_addl.csv')

# Team Stats for 2023 only
team_stats_ratios_2023 = team_stats_ratios[(team_stats_ratios['Season'] == 2023)]
team_stats_ratios_2023.to_csv('team_stats_ratios_2023_addl.csv')


# -------------------------
#  Game Descriptive Stats- Avg Ranking for winning and losing teams, winner's location, Plus Wins and Losses, Important Ratios, Per Season
# -------------------------
# Winning Team seasonal metrics- Add to Game List
games_with_season_stats = pd.merge(games[['Season', 'WTeamID', 'LTeamID', 'WLoc']], team_stats_ratios, how = 'left', left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'])
del games_with_season_stats['TeamID']

games_with_season_stats.rename(columns = {'Wins': 'WTeam_Wins',
                                          'Losses': 'WTeam_Losses',
                                          'AvgRank': 'WTeam_AvgRank',
                                          'BestRank': 'WTeam_BestRank',
                                          'WorstRank': 'WTeam_WorstRank',
                                          'WinPct': 'WTeam_WinPct',
                                          'FG_ratio' : 'WTeam_FG_ratio',
                                          '3PT_ratio': 'WTeam_3PT_ratio',
                                          'ATO_ratio': 'WTeam_ATO_ratio'}, inplace = True)

# Losing Team seasonal metrics- Add to Game List
games_with_season_stats = pd.merge(games_with_season_stats, team_stats_ratios, how = 'left', left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'])
del games_with_season_stats['TeamID']

games_with_season_stats.rename(columns = {'Wins': 'LTeam_Wins',
                                          'Losses': 'LTeam_Losses',
                                          'AvgRank': 'LTeam_AvgRank',
                                          'BestRank': 'LTeam_BestRank',
                                          'WorstRank': 'LTeam_WorstRank',
                                          'WinPct': 'LTeam_WinPct',
                                          'FG_ratio' : 'LTeam_FG_ratio',
                                          '3PT_ratio': 'LTeam_3PT_ratio',
                                          'ATO_ratio': 'LTeam_ATO_ratio'}, inplace = True)


# ------------
# Recode the Winner's Location Variable (WLoc)
# ------------
games_with_season_stats['WLoc'] = games_with_season_stats['WLoc'].replace(['N', 'H', 'A'], [0, 1, -1])


# ------------
# Final All-Game File
# ------------
games_with_season_stats.to_csv('games_with_season_stats_addl.csv')




