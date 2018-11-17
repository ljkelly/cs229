################################################################################
# CS 229 Project
# Stanford University
# Fall 2018
# Paavani Dua (paavanid), Liam Kelly (kellylj) and Matthew Lee (mattskl)
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def clean_batting_dataset(csv_path='../baseballdatabank/Batting.csv'):
    # attributes are: playerID,yearID,stint,teamID,lgID,G,AB,R,H,2B,3B,HR,RBI,
    # SB,CS,BB,SO,IBB,HBP,SH,SF,GIDP

    # Load features and labels
    batting = pd.read_csv(csv_path)

    # Filter from 1960 onwards
    batting = batting[batting['yearID'] >= 1960]

    # Get rid of unnecessary fields
    attr_subset = ['playerID', 'yearID', 'G', 'AB', 'R', 'H', '2B', '3B',
                   'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH',
                   'SF', 'GIDP']
    batting = batting[attr_subset]

    # Get the unique players
    players = batting['playerID'].unique()

    player_data = dict()
    for player in players:
        data = batting[batting['playerID'] == player]
        if (data.shape[0] >= 3):
            player_data[player] = data

    # normalize the yearIDs
    # TODO: normalize each field by number of games played
    # TODO: for players who switch teams (i.e. same yearID) add together stats
    for player, data in player_data.items():
        first_year = data['yearID'].min()
        data['yearID'] -= (first_year - 1)

    # TODO: split data into train/test sets

    # Once data is split, return them separately
    return player_data

# TODO: write function to save off modified dataset appropriately

# TODO: write function/class to train 

if __name__ == '__main__':
    np.random.seed(229)
    data_path = os.path.join('..', 'baseballdatabank', 'Batting.csv')
    player_data = clean_batting_dataset(data_path)
    print(player_data)
