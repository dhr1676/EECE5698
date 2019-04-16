# coding=utf-8
import argparse
import time
from math import sqrt
from operator import itemgetter, add
from collections import defaultdict

# import numpy as np
# import pandas as pd
from pyspark import SparkContext

N_SIM_USER = 20
N_REC_MOVIE = 10


def read_data(file_path, sparkContext):
    """
    :param file_path:
    :param sparkContext:
    :return: RDD(userID, movieID, rating)
    """
    data_rdd = sparkContext.textFile(file_path, use_unicode=False) \
        .map(lambda line: line.strip()) \
        .map(lambda line: line.split(",")) \
        .map(lambda line: (int(line[0]), int(line[1])))

    (train_rdd, test_rdd) = data_rdd.randomSplit(weights=[0.75, 0.25], seed=0)
    return train_rdd, test_rdd


def calc_user_sim(train_rdd):
    # 建立电影-用户倒排表
    movie2users = train_rdd \
        .map(lambda (user, movie): (movie, user)) \
        .groupByKey(numPartitions=40) \
        .map(lambda (movie, user_list): (movie, [u for u in user_list]))

    # 记录一下popularity
    movie_popular = movie2users \
        .map(lambda (movie, user_list): (movie, len(user_list)))

    train_movie_count = movie2users.count()

    # C[u][v]
    user_co_related_matrix = movie2users \
        .map(lambda (movie, user_list): get_user_sim_matrix(movie, user_list)) \
        .flatMap(lambda uv_list: uv_list) \
        .map(lambda (u, v): ((u, v), 1)) \
        .reduceByKey(add, numPartitions=40)

    # N[u]
    view_num_map = train_rdd \
        .map(lambda (user, movie): (user, 1)) \
        .reduceByKey(add, numPartitions=40) \
        .collectAsMap()

    # 计算得到用户相似度矩阵 W[u][v]: RDD((u, v), score)
    user_sim_matrix = user_co_related_matrix \
        .map(lambda ((u, v), count): ((u, v), count / sqrt(view_num_map[u] * view_num_map[v])))
    return user_sim_matrix


def get_user_sim_matrix(movie, user_list):
    uv_list = []
    user_list.sort()
    for u in user_list:
        for v in user_list:
            if u == v:
                continue
            uv_list.append((u, v))
    return uv_list


def recommend(user, watched_movies, _user_sim_matrix_map, other_user_history):
    K = N_SIM_USER
    N = N_REC_MOVIE

    rank = {}

    sort_user_list = sorted(_user_sim_matrix_map.items(), key=lambda x: x[1], reverse=True)[:K]
    for similar_user, similarity_factor in sort_user_list:
        for movie in other_user_history[similar_user]:
            if movie in watched_movies:
                continue
            rank.setdefault(movie, 0)
            rank[movie] += similarity_factor

    return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:N]


def evaluate(train_rdd, test_rdd, user_sim_matrix_rdd):
    N = N_REC_MOVIE

    #  varables for precision and recall
    # hit = 0
    # rec_count = 0
    # test_count = 0

    # variables for coverage
    all_rec_movies = set()

    # variables for popularity
    popular_sum = 0

    test_user_movie = test_rdd \
        .groupByKey(numPartitions=40) \
        .map(lambda (user, movie_list): (user, set([m for m in movie_list])))

    test_u_m_map = test_user_movie.collectAsMap()

    user_sim_matrix_map = user_sim_matrix_rdd \
        .map(lambda ((u, v), score): (u, (v, score))) \
        .groupByKey(numPartitions=40) \
        .map(lambda (u, v_set_list): (u, {v_set[0]: v_set[1] for v_set in v_set_list})) \
        .collectAsMap()

    train_user_movie = train_rdd \
        .groupByKey(numPartitions=40) \
        .map(lambda (user, movie_list): (user, [m for m in movie_list]))

    train_user_history = train_user_movie.collectAsMap()

    train_user_list = train_user_movie \
        .map(lambda (user, movie_list):
             (user, recommend(user, movie_list, user_sim_matrix_map[user], train_user_history))) \
        .map(lambda (user, recommend_dict): (user, calc_hit(recommend_dict, test_u_m_map.get(user, {}), N))) \
        .map(lambda (user, (hit, rec_count, test_count)): (hit, rec_count, test_count))

    _hit, _rec_count, _test_count = 0, 0, 0

    ans = train_user_list.collect()
    for i in ans:
        _hit += i[0]
        _rec_count += i[1]
        _test_count += i[2]

    precision = _hit / (1.0 * _rec_count)
    recall = _hit / (1.0 * _test_count)
    print('precision=%.4f, recall=%.4f' % (precision, recall))


def calc_hit(recommend_dict, test_movie_list, N):
    hit = 0
    for movie, score in recommend_dict:
        if movie in test_movie_list:
            hit += 1
    return hit, N, len(test_movie_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UserCF Spark',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_sim_movie')
    parser.add_argument('--n_rec_movie')
    parser.add_argument('--input', default=None, help='Input Data')
    parser.add_argument('--master', default="local[20]", help="Spark Master")

    args = parser.parse_args()
    sc = SparkContext(args.master, 'UserCF Spark')

    train_set, test_set = read_data(file_path=args.input, sparkContext=sc)
    user_similarity_matrix = calc_user_sim(train_rdd=train_set)
    evaluate(train_set, test_set, user_similarity_matrix)
