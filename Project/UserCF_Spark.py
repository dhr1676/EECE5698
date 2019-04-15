# coding=utf-8
import argparse
import time
from operator import itemgetter
from collections import defaultdict

# import numpy as np
# import pandas as pd
from pyspark import SparkContext


def read_data(file_path, sparkContext):
    """
    :param file_path:
    :param sparkContext:
    :return: RDD(userID, movieID, rating)
    """
    data_rdd = sparkContext.textFile(file_path, use_unicode=False) \
        .map(lambda line: line.strip()) \
        .map(lambda line: line.split(",")) \
        .map(lambda line: (line[0], line[1], line[2]))

    (train_rdd, test_rdd) = data_rdd.randomSplit(weights=[0.75, 0.25], seed=0)
    return train_rdd, test_rdd


def calc_user_sim(train_rdd, test_rdd):
    # 建立电影-用户倒排表
    movie2users = train_rdd \
        .map(lambda (user, movie, rating): (movie, user)) \
        .groupByKey() \
        .map(lambda (movie, user_list): (movie, [u for u in user_list]))

    movie_popular = movie2users\
        .map(lambda (movie, user_list): (movie, len(user_list)))


class UserCF:
    def __init__(self):
        self.train_set = {}
        self.test_set = {}

        self.n_sim_user = 20
        self.n_rec_movie = 10

        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UserCF Spark',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_sim_movie')
    parser.add_argument('--n_rec_movie')
    parser.add_argument('--input', default=None, help='Input Data')
    parser.add_argument('--master', default="local[20]", help="Spark Master")

    args = parser.parse_args()
    sc = SparkContext(args.master, 'Graph')

    userCF = UserCF()
    train_set, test_set = read_data(args.input, sc)
