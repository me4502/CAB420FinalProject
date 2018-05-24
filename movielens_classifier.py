import csv

import numpy as np
import pandas as pd
import sklearn

from typing import Tuple, List, Any


def readFile(path: str) -> Tuple[List[str], List[List[Any]]]:
    with open(path, 'r') as f:
        lines: List[List[Any]] = list(csv.reader(f))
        return lines[0], lines[1:]


def makeMovies():
    headers, data = readFile('./dataset/movies.csv')
    movie_df = pd.DataFrame(data, columns=headers)
    return movie_df


def makeRatings():
    headers, data = readFile('./dataset/ratings.csv')
    ratings_df = pd.DataFrame(data, columns=headers)
    ratings_df = ratings_df.drop('timestamp', axis=1)
    return ratings_df


def makeTags():
    headers, data = readFile('./dataset/tags.csv')
    tags_df = pd.DataFrame(data, columns=headers)
    tags_df = tags_df.drop('timestamp', axis=1)
    return tags_df


if __name__ == '__main__':
    print(makeMovies())
    print(makeRatings())
    print(makeTags())
