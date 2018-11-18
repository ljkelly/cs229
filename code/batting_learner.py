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
import json

def clean_batting_dataset(attr_subset, csv_path='../baseballdatabank/Batting.csv'):
    # attributes are: playerID,yearID,stint,teamID,lgID,G,AB,R,H,2B,3B,HR,RBI,
    # SB,CS,BB,SO,IBB,HBP,SH,SF,GIDP

    # Load features and labels
    batting = pd.read_csv(csv_path)

    # Get appropriate subset of data
    batting = batting[attr_subset]

    # Filter from 1960 onwards
    batting = batting[batting['yearID'] >= 1960]

    # For players who switch teams (i.e. same yearID) add stats together
    batting = batting.groupby(['playerID','yearID']).sum()
    batting = batting.reset_index()

    # normalize the features by number of games    
    normalize_features = attr_subset[3:]
    feature_vars = [var+'_N' for var in normalize_features]

    for attr in normalize_features:
        batting[attr+'_N'] = batting.apply(lambda data: data[attr]/data['G'], axis=1)

    # Get the unique players and store their subsets of data in a dict
    players = batting['playerID'].unique()

    # Want players with at least 4 years so we can train on the first 3 years
    # worth of data and then have at least 4 years to say which is their
    # best year
    player_data = dict()
    for player in players:
        data = batting[batting['playerID'] == player]
        if (data.shape[0] >= 4):
            player_data[player] = data.copy()

    # normalize the yearIDs
    for player, data in player_data.items():
        first_year = data['yearID'].min()
        data['leagueYear'] = data['yearID'] - (first_year - 1)        

    # Once data is split, return them separately
    return player_data

# function to split data into train/test sets
def split_data(data_dict, frac_train=0.9):
    keys_list = list(data_dict.keys())
    np.random.shuffle(keys_list)
    data_size = len(keys_list)

    train_set = dict()
    test_set = dict()

    train_size = frac_train * data_size
    count = 0
    # for key, value in data_list:
    for key in keys_list:
        if count < train_size:
            train_set[key] = data_dict[key]
        else:
            test_set[key] = data_dict[key]
        count += 1

    return train_set, test_set

# functions to save off and load modified dataset appropriately
def write_json(data_dict, json_path):
    copy_dict = dict()
    for key, value in data_dict.items():
        copy_dict[key] = value.to_dict()

    with open(json_path, 'w') as fp:
        json.dump(copy_dict, fp)

def read_json(json_path):
    with open(json_path, 'r') as fp:
        copy_dict = json.load(fp)
    data_dict = dict()
    for key, value in copy_dict:
        data_dict[key] = pd.DataFrame.from_dict(value)

    return data_dict

# TODO: write function/class to train 

if __name__ == '__main__':
    np.random.seed(229)
    data_path = os.path.join('..', 'baseballdatabank', 'Batting.csv')

    # Get rid of unnecessary fields and add normalized fields
    attr_subset = ['playerID', 'yearID', 'G', 'AB', 'R', 'H', '2B', '3B',
                   'HR', 'RBI', 'SB', 'CS', 'BB', 'SO']
    player_data = clean_batting_dataset(attr_subset, data_path)
    train, test = split_data(player_data)

    write_json(train, '../data/batting_train.json')
    write_json(test, '../data/batting_test.json')
