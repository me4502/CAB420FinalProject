import csv
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble

#
# Notes for report:
# Started with SVM w/ classifier - was wayyyy too slow
# Switched to random forest. Was quik boi but low accuracy (31.6%)
# Added gradient boosting tests as well
# - Slightly better results, but slower (33.3%)
# Adding genre to the model.
#


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
    with open(path, 'r') as f:
        lines = list(csv.reader(f))
        return lines[0], lines[1:]


genre_map = dict()
genre_count = 0


def genre_mapper(x):
    if x not in genre_map:
        global genre_count
        genre_map[x] = genre_count
        genre_count += 1
    return genre_map[x]


def make_movielens_df():
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
        # (random_forest_classifier, 'Random Forest'): [
        #     [2, 10, 50, 100, 250],  # n_estimators
        #     [2, 4, 8, 15, 30],  # min_samples_split
        #     [1, 4, 8, 15, 30],  # min_samples_leaf
        # ],
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
    print(best_match)
    print(best_match_percentage)


if __name__ == '__main__':
    separate_train_test(make_movielens_df())
