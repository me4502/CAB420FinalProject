import asyncio
import csv
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn import ensemble

#
# Notes for report:
# Started with SVM w/ classifier - was wayyyy too slow
# Switched to random forest. Was quik boi but low accuracy (31.6%)
# Added gradient boosting tests as well
# - Slightly better results, but slower (33.3%)
# Adding genre to the model.
# Looking into general parameters for different data set sizes.
# Optimiser returned [1000,15,7,10,0.1,0.0,0.0,'Gradient Boosting'] at 69%
# Switching to RMSE with cross val -
#   [100,15,7,10,0.1,0.0,0.0,'Gradient Boosting']
#   Average RMSE of 0.86 across the cross validation sets, accuracy of 72%
#   After testing 100 cases of shuffled sets pre-split,
#   the 72% accuracy remains - meaning unlikely to be overfitted
#   random forest RMSE was 1.09

""" Global variables. """
genre_map = dict()
genre_count = 0


# Adapted from https://stackoverflow.com/a/40449726
def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{
            col: np.concatenate(df[col].values) for col in lst_cols
        }).loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{
            col: np.concatenate(df[col].values) for col in lst_cols
        }).append(df.loc[lens == 0, idx_cols]).fillna(
            fill_value
        ).loc[:, df.columns]


def read_file(path):
    with open(path, 'r', encoding='utf8') as f:
        lines = list(csv.reader(f))
        return lines[0], lines[1:]


def genre_mapper(x):
    """ Maps movie genres to the global variable "genre_map"

    Returns:
        Dictionary -- Maps genre id (int) to genre (string)
    """
    if x not in genre_map:
        global genre_count
        genre_map[x] = genre_count
        genre_count += 1
    return genre_map[x]


def make_movielens_df():
    """ Constructs and returns a panda DataFrame object with the following 
        table structure:

                userId movieId rating  genre
        0          1      31    2.5      0
        1          1    1953    4.0     13
        2          1    1953    4.0      4
    
    Returns:
        DataFrame -- panda DataFrame containing usable features.
    """
    headers, data = read_file('./dataset/movies.csv')
    movie_df = pd.DataFrame(data, columns=headers).drop('title', axis=1)
    headers, data = read_file('./dataset/ratings.csv')
    ratings_df = pd.DataFrame(data, columns=headers).drop('timestamp', axis=1)

    combined_df = ratings_df.merge(movie_df, how='left', on='movieId')
    combined_df = combined_df.assign(**{
        'genres': combined_df['genres'].str.split('|')
    })
    combined_df = explode(combined_df, 'genres')
    combined_df['genre'] = combined_df['genres'].apply(genre_mapper)
    return combined_df.drop('genres', axis=1).sort_values(by='userId')


# linear and rbf (default) kernels used
def svm_classifier(kernel):
    return ensemble.svm.SVC(kernel=kernel)


def random_forest_classifier(n_estimators, min_samples_split, min_samples_leaf):
    return ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )


def gradient_boosting_classifier(
        n_estimators, min_samples_split, min_samples_leaf, max_depth,
        learning_rate, min_weight_fraction_leaf, min_impurity_decrease
):
    return ensemble.GradientBoostingClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        min_impurity_decrease=min_impurity_decrease
    )


def train_model(movielens_df: pd.DataFrame):
    # Use movielens_df.sample(frac=1) for random shuffle
    train, test = train_test_split(movielens_df, test_size=0.3)
    # Parameters determined by running optimiser
    if os.path.isfile('model_cache.pkl'):
        model = joblib.load('model_cache.pkl')
    else:
        model = gradient_boosting_classifier(100, 15, 7, 10, 0.1, 0.0, 0.0)
        model.fit(train.loc[:, train.columns != 'rating'], train['rating'])
        joblib.dump(model, 'model_cache.pkl')
    test_pred = model.predict(test.loc[:, test.columns != 'rating'])

    rmse = math.sqrt(mean_squared_error(test['rating'].values, test_pred))
    print(rmse)


async def _test_algorithm(train_set, test_set, prod, classifier):
    model = classifier(*prod)
    model.fit(train_set.loc[:, train_set.columns != 'rating'],
              train_set['rating'])
    test_pred = model.predict(
        test_set.loc[:, test_set.columns != 'rating']
    )
    test_pred_df = pd.DataFrame(test_pred, columns=['Predicted'])
    test_pred_df['Actual'] = test_set['rating'].reset_index(
        drop=True
    )
    test_pred_df['Same'] = test_pred_df.apply(
        lambda row: row['Predicted'] == row['Actual'], axis=1
    )

    return (
        test_pred_df['Same'].sum() / test_pred_df['Same'].count(),
        math.sqrt(mean_squared_error(
                test_pred_df['Actual'].values,
                test_pred_df['Predicted'].values
        ))
    )


def run_optimiser(movielens_df: pd.DataFrame):
    print('Finding Optimal Parameter Set')
    print('-----------------------------')
    loop = asyncio.get_event_loop()
    kfold = KFold(n_splits=4)

    data_sets = [
        (movielens_df.iloc[train_split], movielens_df.iloc[test_split])
        for train_split, test_split in kfold.split(movielens_df)
    ]
    optimising_parameters = {
        (random_forest_classifier, 'Random Forest'): [
            [10, 50, 100, 250, 500],  # n_estimators
            [3, 7, 15],  # min_samples_split
            [3, 7, 15],  # min_samples_leaf
        ],
        (gradient_boosting_classifier, 'Gradient Boosting'): [
            [100, 500, 1000],  # n_estimators
            [7, 15],  # min_samples_split
            [7, 15],  # min_samples_leaf
            [3, 10],  # max_depth
            [0.1],  # learning_rate
            [0.0],  # min_weight_fraction_leaf
            [0.0],  # min_impurity_decrease
        ]
    }

    best_match = None
    best_match_rmse = 10000000  # Very high value
    best_match_perc = 0

    for classifier, parameters in optimising_parameters.items():
        for prod in itertools.product(*parameters):
            current_perc = 0
            current_rmse = 0
            done, _ = loop.run_until_complete(asyncio.wait([
                _test_algorithm(train_set, test_set, prod, classifier[0])
                for train_set, test_set in data_sets
            ]))
            for future in done:
                result = future.result()
                current_perc += result[0]
                current_rmse += result[1]
            perc = current_perc / len(data_sets)
            rmse = current_rmse / len(data_sets)
            if rmse < best_match_rmse:
                best_match_rmse = rmse
                best_match_perc = perc
                best_match = list(prod) + [classifier[1]]
                print("Current Parameters: {}".format(best_match))
                print("Current Exact Percentage: {}".format(perc))
                print("Current RMSE: {}".format(rmse))
                print("---------------------------------------------------")

    print("Best Parameters: {}".format(best_match))
    print("Best RMSE: {}".format(best_match_rmse))
    print("Best Exact Percentage: {}".format(best_match_perc))
    loop.close()


def explore_data(dataset):
    # Display count for each genre of movie
    unique_genres = dataset.groupby('genre')['movieId'].nunique()
    genre_map_sorted = sorted(genre_map.items(), key=lambda x: x[1])
    genres = [x[0] for x in genre_map_sorted]
    x = dataset['genre']
    num_bins = len(unique_genres)
    plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.xticks(range(num_bins), genres, rotation=90)
    plt.xlabel('Genre')
    plt.ylabel('Review Count')
    plt.title('Histogram of Total Reviews Per Genre')
    plt.show()

    # Plot rating and genre
    rating = dataset.loc[:, 'rating']
    genre = dataset.loc[:, 'genre']
    # totals = np.zeros((20,11))
    totals = np.ones((20, 11))  # Use if normalising with the LogNorm function

    # Sum ratings for each genre for 0-5 stars
    for entry in range(len(rating)):
        totals[int(genre[entry])][int(float(rating[entry])*2)] += 1

    # Lables for axis
    df = pd.DataFrame(totals, index=genres, columns=[x / 2 for x in range(11)])
    plt.pcolor(df, norm=colors.LogNorm())
    # plt.pcolor(df, norm=colors.Normalize())
    # plt.pcolor(df, vmin=0, vmax=3000)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.xlabel('Review Score (Stars)')
    plt.ylabel('Genre')
    plt.title('Heatmap of Review Scores Against Genre')
    plt.show()


if __name__ == '__main__':
    dataset = make_movielens_df()
    if len(sys.argv) > 1 and sys.argv[1] == 'optimise':
        run_optimiser(dataset)
    else:
        explore_data(dataset)
        train_model(dataset)
