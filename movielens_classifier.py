import csv

import numpy as np
import pandas as pd
import sklearn

from typing import Tuple, List, Any


def read_file(path: str) -> Tuple[List[str], List[List[Any]]]:
    with open(path, 'r') as f:
        lines: List[List[Any]] = list(csv.reader(f))
        return lines[0], lines[1:]


def make_movielens_df() -> pd.DataFrame:
    headers, data = read_file('./dataset/movies.csv')
    movie_df = pd.DataFrame(data, columns=headers)
    headers, data = read_file('./dataset/ratings.csv')
    ratings_df = pd.DataFrame(data, columns=headers)

    combined_df = ratings_df.merge(movie_df, how='left', on='movieId')
    combined_df = combined_df.drop(['movieId', 'timestamp', 'genres'], axis=1)
    return combined_df.sort_values(by='userId', axis=1)


if __name__ == '__main__':
    print(make_movielens_df())
