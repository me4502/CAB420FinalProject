import csv
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble

from typing import Tuple, List, Any

#
# Notes for report:
# Started with SVM w/ classifier - was wayyyy too slow
# Switched to random forest. Was quik boi but low accuracy (32%)
# Added gradient boosting tests as well
#


def read_file(path):
    with open(path, 'r') as f:
        lines = list(csv.reader(f))
        return lines[0], lines[1:]


def make_movielens_df():
    headers, data = read_file('./dataset/movies.csv')
    movie_df = pd.DataFrame(data, columns=headers)
    headers, data = read_file('./dataset/ratings.csv')
    ratings_df = pd.DataFrame(data, columns=headers).drop('timestamp', axis=1)

    # combined_df = ratings_df.merge(movie_df, how='left', on='movieId')
    # combined_df = combined_df.drop(['movieId', 'timestamp', 'genres'], axis=1)
    return ratings_df.sort_values(by='userId')


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


def separate_train_test(movielens_df: pd.DataFrame):
    train, test = train_test_split(movielens_df, test_size=0.3)
    optimising_parameters = {
        (random_forest_classifier, 'Random Forest'): [
            [2, 10, 50, 100, 250],  # n_estimators
            [2, 4, 8, 15, 30],  # min_samples_split
            [1, 4, 8, 15, 30],  # min_samples_leaf
        ],
        (gradient_boosting_classifier, 'Gradient Boosting'): [
            [100, 500, 1000, 10000],  # n_estimators
            [2, 4, 8, 15, 30],  # min_samples_split
            [1, 4, 8, 15, 30],  # min_samples_leaf
            [3, 5, 10, 15],  # max_depth
            [0.1, 0.5, 1.0],  # learning_rate
            [0.0, 0.5, 1.0],  # min_weight_fraction_leaf
            [0.0, 0.5, 1.0],  # min_impurity_decrease
        ]
    }

    best_match = None
    best_match_percentage = 0

    for classifier, parameters in optimising_parameters.items():
        for prod in itertools.product(*parameters):
            model = classifier[0](*prod)
            model.fit(train.loc[:, train.columns != 'rating'], train['rating'])
            test_pred = model.predict(test.loc[:, test.columns != 'rating'])
            test_pred_df = pd.DataFrame(test_pred, columns=['Predicted'])
            test_pred_df['Actual'] = test['rating'].reset_index(drop=True)
            test_pred_df['Same'] = test_pred_df.apply(lambda row: row['Predicted'] == row['Actual'], axis=1)
            perc = test_pred_df['Same'].sum() / test_pred_df['Same'].count()
            if perc > best_match_percentage:
                best_match_percentage = perc
                best_match = list(prod) + [classifier[1]]
                print(perc)
    print(best_match)
    print(best_match_percentage)


if __name__ == '__main__':
    separate_train_test(make_movielens_df())
