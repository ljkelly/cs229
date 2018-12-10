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
import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import random

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
        first_year = data['yearID'].min()
        data['leagueYear'] = data['yearID'] - (first_year - 1)

        # # TODO: Fill in gaps in career
        # final_year = data['yearID'].max()
        # career_length = final_year-first_year + 1
        # if (data.shape[0] != career_length):
        #     print("OOps!", player, data)
        #     break

        career_length = data.shape[0]
        # Career length is one of the features we want to estimate.
        data['careerLen'] = career_length

    # Once data is split, return them separately
    return player_data

# function to split data into train/test sets
def split_data(data_dict, frac_train=0.9):
    """Split a python dictionary into train and test datasets

    Args:
        data_dict: a python dictionary you want to split
        frac_train: the fraction of data wanted in the training dataset

    Returns:
        train_set: the data captured by frac_train
        test_set: the data captured by (1-frac_train)
    """
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
    """Save a dictionary of pandas DataFrames to a json file

    Args:
        data_dict: a dictionary of pandas DataFrams
        json_path: the path to the file being saved to
    """
    copy_dict = dict()
    for key, value in data_dict.items():
        copy_dict[key] = value.to_dict()

    with open(json_path, 'w') as fp:
        json.dump(copy_dict, fp)

def read_json(json_path):
    """Load a json dataset that is dict of pandas DataFrames

    Args:
        json_path: the path to the file being read from

    Returns:
        data_dict: a dictionary of pandas DataFrams
    """
    with open(json_path, 'r') as fp:
        copy_dict = json.load(fp)
    data_dict = dict()
    for key, value in copy_dict.items():
        data_dict[key] = pd.DataFrame.from_dict(value)

    return data_dict


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


def make_labels_and_feature_matrix(dataset):
    """ Take dataset in dictionary form and create
        feature matrix and label vector from it.

        Each year's performance for the same player is
        treated independently.
    """
    # Determine number of player data entries,
    # where each year of data is counted independently.
    tot_entries = 0
    for key in dataset.keys():
        tot_entries += len(dataset[key])

    # Keep just the # of games, normalized stats, and year in league
    subattr = [2] + list(range(14,26))

    x = np.empty((tot_entries, len(subattr)))
    y = np.empty(tot_entries)

    i = 0
    for key in dataset.keys():
        ds = dataset[key].values
        for year in range(ds.shape[0]):
            x[i,:] = ds[year,subattr]
            y[i] = ds[-1,-1]    # career length
            i += 1

    return x, y

def make_triplet_feature_matrix(dataset, feature_set):
    """ Take dataset in dictionary form and create feature matrix and label
        vector from it, with each row being the first three years features
        in feature_set.
    """
    num_rows = len(dataset.keys())
    num_features = len(feature_set) - 1
    num_cols = 3 * num_features

    x = np.zeros((num_rows, num_cols))
    y = np.zeros(num_rows)

    i = 0
    for key in dataset.keys():
        ds = dataset[key].values
        for year in range(3):
            start_idx = year * num_features
            end_idx = start_idx + num_features
            x[i, start_idx:end_idx] = ds[year, feature_set[:-1]]
            y[i] = ds[-1,-1] # career length
        i += 1

    return x, y

# Turn player data into <career_length x 1 x num_features>
def prep_lstm_input_tensor(player_data, max_years=100):
    # Keep just the # of games and normalized stats
    subattr = [2] + list(range(14,25))
    n_features = len(subattr)
    # Player data sometimes not ordered by increasing year..
    true_career_len = np.amax(player_data.values[:, -2])
    # career_len = min(max_years, true_career_len)

    # # Zero-pad stats to 20 years so LTSM doens't just learn # of input steps==>output during training
    # career_len = max(20, true_career_len)
    # # Just use 10 years of data?
    # career_len = 10

    career_len = max(max_years, 6)

    # # Produce truncated, non-zero-padded tensors for prediction phase.
    # career_len = min(career_len, max_years)

    tensor = torch.zeros(career_len, 1, n_features)

    # If didn't play in a year, stats are 0
    for i in range(player_data.shape[0]):
        year = player_data.values[i, -2]
        if (year > max_years or year > career_len):
            break
        tensor[year-1][0][:] = torch.from_numpy(player_data.values[i, subattr].astype(float))

    return true_career_len, tensor


def linear_test(train, test):
    """Train a linear model, test it, plot the results

    Args:
        train: data dictionary to train the linear model on
        test: data dictionary to test against
    """
    x_train, y_train = make_labels_and_feature_matrix(train)
    x_test, y_test = make_labels_and_feature_matrix(test)

    reg = LinearRegression().fit(x_train, y_train)
    # print(reg.score(x_train, y_train))
    # print(reg.coef_)
    # print(reg.intercept_)
    y_pred = reg.predict(x_test)
    deltas = y_pred-y_test
    np.savetxt("./output/linear_prediction.txt", y_pred)
    np.savetxt("./output/linear_labels.txt", y_test)
    np.savetxt("./output/linear_deltas.txt", deltas)
    print('Average error: ', np.mean(abs(y_pred - y_test)))

    poly = np.polyfit(y_test, y_pred, 1)
    y_reg = poly[0] * y_test + poly[1]
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_reg, 'r')
    plt.xlabel('True number of years in the league')
    plt.ylabel('Predicted number of years in the league')
    plt.title('Linear Regression data comparison')
    plt.savefig('output/linear_reg_output.png')
    plt.clf()

def three_year_test(train, test):
    """Train a linear model, test it, plot the results

    Args:
        train: data dictionary to train the linear model on
        test: data dictionary to test against
    """
    subattr = [2] + list(range(14,26))
    x_train, y_train = make_triplet_feature_matrix(train, subattr)
    x_test, y_test = make_triplet_feature_matrix(test, subattr)

    reg = LinearRegression().fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    deltas = y_pred-y_test
    # print(y_pred)
    # print(y_test)
    np.savetxt("./output/triplet_prediction.txt", y_pred)
    np.savetxt("./output/triplet_labels.txt", y_test)
    np.savetxt("./output/triplet_deltas.txt", deltas)
    print('Average error: ', np.mean(abs(y_pred - y_test)))

    poly = np.polyfit(y_test, y_pred, 1)
    y_reg = poly[0] * y_test + poly[1]
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_reg, 'r')
    plt.xlabel('True number of years in the league')
    plt.ylabel('Predicted number of years in the league')
    plt.title('Linear Regression on three years\' data comparison')
    plt.savefig('output/triplet_reg_output.png')
    plt.clf()


## From http://www.jessicayung.com/lstms-for-time-series-in-pytorch/
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

        # TODO: Relu the output?

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)


def lstm_test(train, test):
    ## Select # hidden based on this: https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model
    n_hidden = 64

    learning_rate = 0.0005
    n_epochs = 50
    n_features = len([2] + list(range(14,25)))
    n_train = len(train.keys())

    tensors = []
    career_lens = []

    # Prep training tensors
    for key in train.keys():
        career_len, p_tensor = prep_lstm_input_tensor(train[key], 3)
        tensors.append(p_tensor)
        career_lens.append(career_len)
    n_tensors = len(tensors)


    # Create model:
    # n_features for each input timestep
    # output single value
    # 1 LSTM layer (and 1 FC layer to convert stats to career length)
    model = LSTM(n_features, n_hidden, batch_size=1, output_dim=1, num_layers=1)

    # Use MSE since we're looking at career length match.
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train LSTM model
    hist = np.zeros(n_epochs)

    ## From http://www.jessicayung.com/lstms-for-time-series-in-pytorch/
    for t in range(n_epochs):
        rr = list(range(n_tensors))
        random.shuffle(rr)
        j = 0
        for i in rr:
            # Clear stored gradient
            model.zero_grad()

            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            model.hidden = model.init_hidden()

            # Forward pass
            y_pred = model(tensors[i])

            loss = loss_fn(y_pred, torch.tensor(float(career_lens[i])))
            hist[t] += loss.item()

            # Zero out gradient, else they will accumulate between epochs/samples
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()

            if j % 1000 == 0:
                print("Epoch ", t, "item ", j, "MSE: ", loss.item())
            j += 1
        hist[t] /= float(n_tensors)
        print("Epoch ", t, "average MSE: ", hist[t])


    test_tensors = []
    test_career_lens = []

    # Prep test tensors - only first 3 years of timeseries.
    for key in test.keys():
        career_len, p_tensor = prep_lstm_input_tensor(test[key], 3)
        test_tensors.append(p_tensor)
        test_career_lens.append(career_len)
    n_test_tensors = len(test_tensors)

    test_preds = []
    for i in range(n_test_tensors):
        # Reset hidden state between predictions
        model.hidden = model.init_hidden()

        # Forward pass
        y_pred = model(test_tensors[i])
        test_preds.append(y_pred.item())

    test_preds = np.asarray(test_preds)
    test_career_lens = np.asarray(test_career_lens)
    deltas = test_preds-test_career_lens

    np.savetxt("./output/lstm_epoch_err.txt", hist)
    np.savetxt("./output/lstm_prediction.txt", test_preds)
    np.savetxt("./output/lstm_labels.txt", test_career_lens)
    np.savetxt("./output/lstm_deltas.txt", deltas)
    print('Average error: ', np.mean(abs(deltas)))

    poly = np.polyfit(test_career_lens, test_preds, 1)
    y_reg = poly[0] * test_career_lens + poly[1]
    plt.scatter(test_career_lens, test_preds)
    plt.plot(test_career_lens, y_reg, 'r')
    plt.xlabel('True number of years in the league')
    plt.ylabel('Predicted number of years in the league')
    plt.title('LSTM data comparison')
    plt.savefig('output/lstm_output.png')


if __name__ == '__main__':
    np.random.seed(229)
    torch.manual_seed(229)
    random.seed(229)

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

    linear_test(train, test)
    three_year_test(train, test)
    lstm_test(train, test)
