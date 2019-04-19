# coding=utf-8
import argparse
import time
from math import sqrt, log
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ItemCF:
    def __init__(self, n_sim_movie=40, n_rec_movie=10):
        self.n_sim_movie = n_sim_movie
        self.n_rec_movie = n_rec_movie

        self.train_set = {}
        self.test_set = {}

        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0

    def read_data(self, file_path):
        """
        Read data from file
        :param file_path:
        :return:
        """
        raw_data = pd.read_csv(file_path, dtype={'userId': str, 'movieId': str})
        print("Read data finished!")
        train_set, test_set = train_test_split(raw_data, test_size=0.25)

        for i in range(len(train_set)):
            user = train_set.iloc[i]["userId"]
            movie = train_set.iloc[i]["movieId"]
            rating = train_set.iloc[i]["rating"]
            self.train_set.setdefault(user, {})
            self.train_set[user][movie] = rating

        for i in range(len(test_set)):
            user = test_set.iloc[i]["userId"]
            movie = test_set.iloc[i]["movieId"]
            rating = test_set.iloc[i]["rating"]
            self.test_set.setdefault(user, {})
            self.test_set[user][movie] = rating
        return

    def calc_movie_sim(self):
        """
        Calculate movie similarity matrix
        :return:
        """
        # Get the movie watch times
        for user, movies in self.train_set.items():
            for movie in movies:
                self.movie_popular[movie] = self.movie_popular.get(movie, 0) + 1

        self.movie_count = len(self.movie_popular)
        print("Total movie number = %d" % self.movie_count)

        # Get the item co-rated matrix
        for user, movies in self.train_set.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    self.movie_sim_matrix[m1][m2] += 1
        print("Build co-rated users matrix success!")

        # Get the item similarity matrix
        print("Calculating movie similarity matrix ...")
        for m1, related_movies in self.movie_sim_matrix.items():
            for m2, count in related_movies.items():
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    self.movie_sim_matrix[m1][m2] = count / np.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
        print('Calculate movie similarity matrix success!')

    def recommend(self, user):
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.train_set[user]
        # print("user ", user, " watched ", watched_movies)
        # user  1  watched  {'1061': '3.0', '1129': '2.0', '1172': '4.0',
        # '1263': '2.0', '1293': '2.0', '1339': '3.5', '1343': '2.0', '1405': '1.0',
        # '1953': '4.0', '2150': '3.0', '2193': '2.0'}

        for movie, rating in watched_movies.items():
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:  # Continue if this movie has bee watched
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += w * float(rating)
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def evaluate(self):
        print('Evaluating start ...')
        N = self.n_rec_movie

        hit = 0
        rec_count = 0
        test_count = 0

        for i, user in enumerate(self.train_set):
            test_movies = self.test_set.get(user, {})
            rec_movies = self.recommend(user)
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        print('precision = %.4f, recall = %.4f' % (precision, recall))


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='ItemCF Sequential',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_sim_movie')
    parser.add_argument('--n_rec_movie')
    parser.add_argument('--input')
    args = parser.parse_args()

    data_path = args.input
    if args.n_sim_movie and args.n_rec_movie:
        itemCF = ItemCF(n_sim_movie=args.n_sim_movie,
                        n_rec_movie=args.n_rec_movie)
    else:
        itemCF = ItemCF()
    itemCF.read_data(data_path)
    itemCF.calc_movie_sim()
    itemCF.evaluate()
    end_time = time.time()
    print("Time elapse: %.2f s\n" % (end_time - start_time))
