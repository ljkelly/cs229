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
import sys
import json
from sklearn.linear_model import LinearRegression
# from sklearn import svm

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
        career_length = data.shape[0]
        first_year = data['yearID'].min()
        data['leagueYear'] = data['yearID'] - (first_year - 1)
        # Career length is one of the features we want to estimate.
        data['careerLen'] = career_length

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
    for key, value in copy_dict.items():
        data_dict[key] = pd.DataFrame.from_dict(value)

    return data_dict

# TODO: write function/class to train

# # Use Fangraph's equation for wins above replacement
# def player_fWAR(attr_subset, player_data):
#     # attributes are: playerID,yearID,stint,teamID,lgID,G,AB,R,H,2B,3B,HR,RBI,
#     # SB,CS,BB,SO,IBB,HBP,SH,SF,GIDP
#     pd = player_data

#     # The coefficients vary by year based on freq of occurance.
#     # wOBA coefficients available here, update later:
#     # https://www.fangraphs.com/guts.aspx?type=cn
#     wOBA_num = 0.69*(pd['BB']-pd['IBB']) + 0.72*pd['HBP'] + 0.88*pd['H']
#     wOBA_num += 1.247*pd['2B'] + 1.578*pd['3B'] + 2.031*pd['HR']
#     wOBA_denom = pd['AB'] + pd['BB'] - pd['IBB'] + pd['SF'] + pd['HBP']
#     wOBA = wOBA_num/wOBA_denom

#     # League average for specific year (fix to 2018 for now)
#     wOBA_league = 0.315
#     # Plate appearance ~= at bet?
#     wRAA = (wOBA-lgwOBA)/wOBAScale * AB
#     war = pd[]

# def linReg()

def make_labels_and_feature_matrix(dataset):
    """ Take dataset in dictionary form and create
        feature matrix and label vector from it.

        Each year's performance for the same player is
        treated independently.
    """
    # Determine number of player data entries,
    # where each year of data is counted independently.
    tot_entries = 0
    for key in train.keys():
        tot_entries += len(train[key])

    # Keep just the # of games, normalized stats, and year in league
    subattr = [2] + list(range(14,26))

    x = np.empty((tot_entries, len(subattr)))
    y = np.empty(tot_entries)

    i = 0
    for key in train.keys():
        ds = train[key].values
        for year in range(ds.shape[0]):
            x[i,:] = ds[year,subattr]
            y[i] = ds[year,26]    # career length
            i += 1

    return x, y

if __name__ == '__main__':
    np.random.seed(229)

    reprune = 0 if len(sys.argv) <= 1 else int(sys.argv[1])

    # Get rid of unnecessary fields and add normalized fields
    attr_subset = ['playerID', 'yearID', 'G', 'AB', 'R', 'H', '2B', '3B',
                   'HR', 'RBI', 'SB', 'CS', 'BB', 'SO']
    train = None
    test = None
    if (reprune):
        data_path = os.path.join('..', 'baseballdatabank', 'Batting.csv')

        player_data = clean_batting_dataset(attr_subset, data_path)
        train, test = split_data(player_data)

        write_json(train, '../data/batting_train.json')
        write_json(test, '../data/batting_test.json')
    else:
        train = read_json('../data/batting_train.json')
        test  = read_json('../data/batting_test.json')

    x_train, y_train = make_labels_and_feature_matrix(train)
    x_test, y_test = make_labels_and_feature_matrix(test)

    reg = LinearRegression().fit(x_train, y_train)
    # print(reg.score(x_train, y_train))
    # print(reg.coef_)
    # print(reg.intercept_)
    y_pred = reg.predict(x_test)
    # print(y_pred)
    # print(y_test)
    np.savetxt("./output/prediction.txt", y_pred)
    np.savetxt("./output/labels.txt", y_test)
    np.savetxt("./output/deltas.txt", y_pred-y_test)
    # delta = abs(y_pred - y_test)/len(y_pred)
    # deviation = sum(delta)
    # print(deviation)

