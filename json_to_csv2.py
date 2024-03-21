import pandas as pd
import numpy as np
import json

with open('teams_data.json', 'r') as file:
    data = json.load(file)

teams_data = []
for team_name, details in data.items():
    teams_data.append({
        'Team_Name': team_name.replace(' ', '_'),
        'Position': int(details['position'][1:]),
        'Points': int(details['points'].split()[0]),
        'Win_Percentage': float(details['win_percentage'][:-1]),
    })
df_teams = pd.DataFrame(teams_data)

players_data = []
for team_name, details in data.items():
    for player_name, stats in details['players'].items():
        players_data.append({
            'Team_Name': team_name.replace(' ', '_'),
            'Player_Name': player_name.replace(' ', '_'),
            'Rating': float(stats['Rating']),
            'Kills_Per_Round': float(stats['Kills Per Round']),
            'Deaths_Per_Round': float(stats['Deaths Per Round']),
            'Headshots': float(stats['Headshots'][:-1]),
            'Rounds_Contributed': float(stats['Rounds Contributed'][:-1]),
            'Maps_Played': int(stats['Maps Played']),
        })
df_players = pd.DataFrame(players_data)

player_stats_agg = df_players.groupby('Team_Name').agg({
    'Rating': 'mean',
    'Kills_Per_Round': 'mean',
    'Deaths_Per_Round': 'mean',
    'Headshots': 'mean',
    'Rounds_Contributed': 'mean'
}).reset_index()

player_stats_agg_rounded = player_stats_agg.copy()
player_stats_agg_rounded['Rating'] = player_stats_agg_rounded['Rating'].round(2)
player_stats_agg_rounded['Kills_Per_Round'] = player_stats_agg_rounded['Kills_Per_Round'].round(2)
player_stats_agg_rounded['Deaths_Per_Round'] = player_stats_agg_rounded['Deaths_Per_Round'].round(2)
player_stats_agg_rounded['Headshots'] = player_stats_agg_rounded['Headshots'].round(2)
player_stats_agg_rounded['Rounds_Contributed'] = player_stats_agg_rounded['Rounds_Contributed'].round(2)

matches_data = []
for team_name, details in data.items():
    for match in details['matches']:
        if match['score'] != '-:-':
            team1_name = match['team1'].replace(' ', '_')
            team2_name = match['team2'].replace(' ', '_')

            team1_filter = df_teams[df_teams['Team_Name'] == team1_name]
            team2_filter = df_teams[df_teams['Team_Name'] == team2_name]

            if not team1_filter.empty and not team2_filter.empty:
                team1_stats = team1_filter.iloc[0]
                team2_stats = team2_filter.iloc[0]
                team1_players_stats = player_stats_agg_rounded[player_stats_agg_rounded['Team_Name'] == team1_name].iloc[0]
                team2_players_stats = player_stats_agg_rounded[player_stats_agg_rounded['Team_Name'] == team2_name].iloc[0]

                matches_data.append({
                    'Date': match['date'],
                    'Team1': team1_name,
                    'Team1_Position': team1_stats['Position'],
                    'Team1_Points': team1_stats['Points'],
                    'Team1_Win_Percentage': team1_stats['Win_Percentage'],
                    'Team1_Rating': team1_players_stats['Rating'],
                    'Team1_Kills_Per_Round': team1_players_stats['Kills_Per_Round'],
                    'Team1_Deaths_Per_Round': team1_players_stats['Deaths_Per_Round'],
                    'Team1_Headshots': team1_players_stats['Headshots'],
                    'Team1_Rounds_Contributed': team1_players_stats['Rounds_Contributed'],
                    'Team2': team2_name,
                    'Team2_Position': team2_stats['Position'],
                    'Team2_Points': team2_stats['Points'],
                    'Team2_Win_Percentage': team2_stats['Win_Percentage'],
                    'Team2_Rating': team2_players_stats['Rating'],
                    'Team2_Kills_Per_Round': team2_players_stats['Kills_Per_Round'],
                    'Team2_Deaths_Per_Round': team2_players_stats['Deaths_Per_Round'],
                    'Team2_Headshots': team2_players_stats['Headshots'],
                    'Team2_Rounds_Contributed': team2_players_stats['Rounds_Contributed'],
                    'Score': 1 if int(match['score'].split(':')[0]) > int(match['score'].split(':')[1]) else 0
                })
        

df_matches = pd.DataFrame(matches_data)

df_matches.to_csv('matches_data.csv', index=False)
