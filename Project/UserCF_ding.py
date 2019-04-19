# coding=utf-8
import argparse
import time
from operator import itemgetter
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class UserCF:
    def __init__(self):
        self.train_set = {}
        self.test_set = {}

        self.num_users = 944
        self.num_movies = 1682
        self.num_k = 20

        self.n_sim_user = 35
        self.n_rec_movie = 10

        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        self.log_file = open(r'./log_file.txt', 'w')

    def read_data(self, file_path):
        raw_data = pd.read_csv(file_path,
                               dtype={'userId': str, 'movieId': str})
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

        # 字典中数据长这样：{642: {'162': '5.0', '457': '2.0'...}, }
        # for k, v in self.trainSet.items():
        #     print(k, v)
        return

    def calc_user_sim(self):
        """ calculate user similarity matrix """
        # build inverse table for item-users
        # key=movieID, value=list of userIDs who have seen this movie
        # print('building movie-users inverse table...', file=sys.stderr)
        print('building movie-users inverse table...')
        movie2users = {}

        for user, movies in self.train_set.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)
                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        print('build movie-users inverse table successful')

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print('total movie number = %d' % self.movie_count)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print('building user co-rated movies matrix...')

        for movie, users in movie2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1
        print('build user co-rated movies matrix succ')

        # calculate similarity matrix
        print('calculating user similarity matrix...')
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / np.sqrt(len(self.train_set[u]) * len(self.train_set[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating user similarity factor(%d)' % simfactor_count)

        print('calculate user similarity matrix(similarity factor) successful')
        print('Total similarity factor number = %d' %
              simfactor_count)

    def recommend(self, user):
        """ Find K similar users and recommend N movies. """
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = dict()
        watched_movies = self.train_set[user]

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.train_set[similar_user]:
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += similarity_factor
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        """
        print evaluation result: precision, recall, coverage and popularity
        """
        print('Evaluation start...')

        N = self.n_rec_movie
        #  variables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # variables for coverage
        all_rec_movies = set()
        # # variables for popularity
        # popular_sum = 0

        for i, user in enumerate(self.train_set):
            if i % 500 == 0:
                print('recommended for %d users' % i)
            test_movies = self.test_set.get(user, {})
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                # popular_sum += np.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        # popularity = popular_sum / (1.0 * rec_count)

        # print('precision = %.4f\t recall = %.4f\t coverage = %.4f\t popularity = %.4f' %
        #       (precision, recall, coverage, popularity))
        print('precision = %.4f\t recall = %.4f\t coverage = %.4f\t' %
              (precision, recall, coverage))


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='UserCF Standalone',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_sim_movie')
    parser.add_argument('--n_rec_movie')
    args = parser.parse_args()

    # data_path = './data/ratings_100w_old.csv'
    data_path = './data/ratings_75k.csv'
    userCF = UserCF()
    userCF.read_data(data_path)
    userCF.calc_user_sim()
    userCF.evaluate()
    userCF.log_file.close()

    end_time = time.time()
    print("Time elapse: %.2f s\n" % (end_time - start_time))
