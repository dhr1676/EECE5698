# coding=utf-8
"""
Created on 2019-04-16

@author: Haoran Ding
"""
import argparse
import time
from math import sqrt
from operator import add

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
        .map(lambda line: (int(line[0]), int(line[1]), float(line[2])))

    (train_rdd, test_rdd) = data_rdd.randomSplit(weights=[0.75, 0.25], seed=0)
    return train_rdd, test_rdd


def calc_item_sim(train_rdd):
    movie_popular = train_rdd \
        .map(lambda (user, movie, rating): (movie, 1)) \
        .reduceByKey(add, numPartitions=40)

    movie_count = movie_popular.count()

    # 物品共现矩阵 C[i][j]
    item_co_related_matrix = train_rdd \
        .map(lambda (user, movie, rating): (user, movie)) \
        .groupByKey(numPartitions=40) \
        .map(lambda (user, movie_list): (user, [m for m in movie_list])) \
        .map(lambda (user, movie_list): get_item_sim_matrix(user, movie_list)) \
        .flatMap(lambda ij_list: ij_list) \
        .map(lambda (i, j): ((i, j), 1)) \
        .reduceByKey(add, numPartitions=40)

    # N[i]
    view_num_map = movie_popular.collectAsMap()
    b_view_num_map = sc.broadcast(view_num_map)

    # 计算得到物品相似度矩阵 W[i][j]: RDD((i, j), score)
    item_sim_matrix = item_co_related_matrix \
        .map(lambda ((i, j), count): ((i, j), count / sqrt(view_num_map[i] * view_num_map[j])))
    # .map(lambda ((i, j), count): ((i, j), count / sqrt(b_view_num_map.value([i]) * b_view_num_map.value([j]))))
    return item_sim_matrix


def get_item_sim_matrix(user, movie_list):
    ij_list = []
    movie_list.sort()
    for i in movie_list:
        for j in movie_list:
            if i == j:
                continue
            ij_list.append((i, j))
    return ij_list


def recommend(watched_movies, _item_sim_matrix_map):
    # watched_movies: {m1: score1, m2: score2, ...}
    # _item_sim_matrix_map: {movieID: {movieID: score}, ...}

    K = N_SIM_USER
    N = N_REC_MOVIE

    rank = {}

    for movie, rating in watched_movies.items():
        sort_item_list = sorted(_item_sim_matrix_map[movie].items(),
                                key=lambda x: x[1], reverse=True)[:K]
        for related_movie, similarity_factor in sort_item_list:
            if related_movie in watched_movies:
                continue
            rank.setdefault(related_movie, 0)
            rank[related_movie] += similarity_factor * rating

    return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:N]


def recommend_new(user, movie, rating, _item_sim_matrix_map_dict):
    K = N_SIM_USER
    N = N_REC_MOVIE
    rank = {}

    sort_item_list = sorted(_item_sim_matrix_map_dict.items(), key=lambda x: x[1], reverse=True)[:K]
    for related_movie, similarity_factor in sort_item_list:
        if related_movie == movie:
            continue
        rank.setdefault(related_movie, 0)
        rank[related_movie] += similarity_factor * rating

    sort_rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:N]
    res = []
    for pair in sort_rank:
        res.append((user, pair[0], pair[1]))
    # res: [(user, m1, score1), (user, m2, score2), ...]
    return res


def recommend_help(recommend_list):
    sort_recommend_list = sorted(recommend_list, key=lambda x: x[1], reverse=True)
    return sort_recommend_list[:N_REC_MOVIE]


def evaluate(train_rdd, test_rdd, item_sim_matrix_rdd):
    N = N_REC_MOVIE

    test_user_movie = test_rdd \
        .map(lambda (user, movie, rating): (user, movie)) \
        .groupByKey(numPartitions=40) \
        .map(lambda (user, movie_list): (user, set([m for m in movie_list])))

    test_u_m_map = test_user_movie \
        .collectAsMap()

    # b_test_u_m_map = sc.broadcast(test_u_m_map)

    item_sim_matrix_map = item_sim_matrix_rdd \
        .map(lambda ((i, j), score): (i, (j, score))) \
        .groupByKey(numPartitions=40) \
        .map(lambda (i, j_set_list): (i, {j_set[0]: j_set[1] for j_set in j_set_list})) \
        .collectAsMap()
    # { movieID: {m1: score, m2: score}, ...}

    b_item_sim_matrix_map = sc.broadcast(item_sim_matrix_map)

    # train_user_movie = train_rdd \
    #     .map(lambda (user, movie, rating): (user, (movie, rating))) \
    #     .groupByKey(numPartitions=40) \
    #     .map(lambda (user, movie_list): (user, {m_set[0]: m_set[1] for m_set in movie_list}))
    # # train_user_movie: RDD( (user, {m1: score1, m2: score2}) )

    # train_user_list = train_user_movie \
    #     .map(lambda (user, movie_list_dict):
    #          (user, recommend(movie_list_dict, b_item_sim_matrix_map.value()))) \
    #     .map(lambda (user, recommend_dict): (user, calc_hit(recommend_dict, test_u_m_map.get(user, {}), N))) \
    #     .map(lambda (user, (hit, rec_count, test_count)): (hit, rec_count, test_count))

    # train_user_list = train_rdd \
    #     .map(lambda (user, movie, rating): recommend_new(user, movie, rating, item_sim_matrix_map[movie])) \
    #     .flatMap(lambda user_recommend: user_recommend) \
    #     .map(lambda (user, movie, score): (user, (movie, score))) \
    #     .groupByKey(numPartitions=40) \
    #     .map(lambda (user, rec_list): (user, [(rec[0], rec[1]) for rec in rec_list])) \
    #     .map(lambda (user, rec_list): (user, recommend_help(rec_list))) \
    #     .map(lambda (user, recommend_dict): (user, calc_hit(recommend_dict, test_u_m_map.get(user, {}), N))) \
    #     .map(lambda (user, (hit, rec_count, test_count)): (hit, rec_count, test_count)) \
    #     .persist()

    train_user_list = train_rdd \
        .map(lambda (user, movie, rating): recommend_new(user, movie, rating, item_sim_matrix_map[movie])) \
        .flatMap(lambda user_recommend: user_recommend) \
        .map(lambda (user, movie, score): (user, (movie, score))) \
        .groupByKey(numPartitions=40) \
        .map(lambda (user, rec_list): (user, recommend_help(rec_list))) \
        .map(lambda (user, recommend_dict): (user, calc_hit(recommend_dict, test_u_m_map.get(user, {}), N))) \
        .persist()

    _hit = train_user_list \
        .map(lambda (user, (hit, rec_count, test_count)): (1, hit)) \
        .reduceByKey(add, numPartitions=40) \
        .collect()
    _rec_count = train_user_list \
        .map(lambda (user, (hit, rec_count, test_count)): (1, rec_count)) \
        .reduceByKey(add, numPartitions=40) \
        .collect()
    _test_count = train_user_list \
        .map(lambda (user, (hit, rec_count, test_count)): (1, test_count)) \
        .reduceByKey(add, numPartitions=40) \
        .collect()

    train_user_list.unpersist()

    print _hit, _rec_count, _test_count

    # _hit, _rec_count, _test_count = 0, 0, 0
    #
    # ans = train_user_list.collect()
    # for i in ans:
    #     _hit += i[0]
    #     _rec_count += i[1]
    #     _test_count += i[2]

    # precision = _hit / (1.0 * _rec_count)
    # recall = _hit / (1.0 * _test_count)
    # print 'precision=%.4f, recall=%.4f' % (precision, recall)

    return


def calc_hit(recommend_dict, test_movie_list, N):
    hit = 0
    for movie, score in recommend_dict:
        if movie in test_movie_list:
            hit += 1
    return hit, N, len(test_movie_list)


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='ItemCF Spark',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_sim_movie')
    parser.add_argument('--n_rec_movie')
    parser.add_argument('--input', default=None, help='Input Data')
    parser.add_argument('--master', default="local[20]", help="Spark Master")

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    sc = SparkContext(args.master, 'ItemCF Spark Version')

    if not args.verbose:
        sc.setLogLevel("ERROR")

    train_set, test_set = read_data(file_path=args.input, sparkContext=sc)
    item_similarity_matrix = calc_item_sim(train_rdd=train_set)
    evaluate(train_set, test_set, item_similarity_matrix)

    end_time = time.time()
    print "Time elapse: %.2f s\n" % (end_time - start_time)

    # python ItemCF_Spark_ding.py --input ./data/ratings_100k_spark.csv
