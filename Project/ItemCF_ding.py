# coding=utf-8
# -*- coding: utf-8 -*-

# 基于项目的协同过滤推荐算法实现
import random
import time

import math
from operator import itemgetter


class ItemBasedCF:
    # 初始化参数
    def __init__(self):
        # 找到相似的20部电影，为目标用户推荐10部电影
        self.n_sim_movie = 20
        self.n_rec_movie = 10

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵
        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0  # 总电影的个数

        print('Similar movie number = %d' % self.n_sim_movie)
        print('Recommneded movie number = %d' % self.n_rec_movie)

    # 读文件得到“用户-电影”数据，建立用户-电影倒排表
    def get_dataset(self, filename, pivot=0.75):
        trainSet_len = 0
        testSet_len = 0
        random.seed(0)
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            if random.random() < pivot:  # 返回随机生成的一个实数，它在[0,1)范围内；程序中75%为训练集
                self.trainSet.setdefault(user, {})  # 如果键不存在于字典中，将会添加键并将值设为默认值
                self.trainSet[user][movie] = rating  # 建立用户-电影倒排表
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1

        # 字典中数据长这样：{642: {'162': '5.0', '457': '2.0'...}, }
        # for k, v in self.trainSet.items():
        #     print(k, v)

        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)

    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')  # yield 关注一下
        print('Load %s success!' % filename)

    # 计算电影之间的相似度
    def calc_movie_sim(self):
        # 得到每个电影的被用户看过的次数
        for user, movies in self.trainSet.items():
            for movie in movies:
                self.movie_popular[movie] = self.movie_popular.get(movie, 0) + 1
                # if movie not in self.movie_popular:
                #     self.movie_popular[movie] = 0
                # self.movie_popular[movie] += 1

        self.movie_count = len(self.movie_popular)
        print("Total movie number = %d" % self.movie_count)

        # 可以得到物品共现矩阵
        for user, movies in self.trainSet.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    self.movie_sim_matrix[m1][m2] += 1
        print("Build co-rated users matrix success!")
        # for k, v in self.movie_sim_matrix.items():
        #     print(k, v)

        # 计算电影之间的相似性
        print("Calculating movie similarity matrix ...")
        for m1, related_movies in self.movie_sim_matrix.items():
            for m2, count in related_movies.items():
                # 注意0向量的处理，即某电影的用户数为0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    # 这是增加惩罚项的公式，但并不是IUF
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
        print('Calculate movie similarity matrix success!')

    # 针对目标用户U，找到K部相似的电影，并推荐其N部电影
    def recommend(self, user):
        K = self.n_sim_movie  # 找K个相似的物品推荐
        N = self.n_rec_movie  # 最终推荐N个物品
        rank = {}
        watched_movies = self.trainSet[user]  # 从train中找到该user看过的电影集合，不存在找不到，因为user就是从train里找的
        # print("user ", user, " watched ", watched_movies)
        # user  1  watched  {'1061': '3.0', '1129': '2.0', '1172': '4.0',
        # '1263': '2.0', '1293': '2.0', '1339': '3.5', '1343': '2.0', '1405': '1.0',
        # '1953': '4.0', '2150': '3.0', '2193': '2.0'}

        for movie, rating in watched_movies.items():
            # [('89', 0.39405520311955033), ('832', 0.3737654899566537), ('1611', 0.34752402342845795),
            # ('830', 0.34403123102809335), ('996', 0.34050261230349943), ('707', 0.3348554112644579),
            # ('4280', 0.3296902366978935), ('195', 0.3296902366978935), ('535', 0.3263766828841098),
            # ('799', 0.3263766828841098), ('1339', 0.32564480451291805), ('1438', 0.3230291412348993),
            # ('1518', 0.3143473067309657), ('1216', 0.3127716210856122), ('3218', 0.3127716210856122),
            # ('4318', 0.3127716210856122), ('4482', 0.3127716210856122), ('1535', 0.3127716210856122),
            # ('1094', 0.3044295411712823), ('1608', 0.30414953233623676)]
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:  # 如果已经出现在看过的电影中，就不加入推荐列表
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += w * float(rating)  # 加入进rank中还有一个打分，
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print('Evaluating start ...')
        N = self.n_rec_movie

        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user in enumerate(self.trainSet):
            test_movies = self.testSet.get(user, {})  # 获得test中的dict，如果没有user这个key，返回一个空dict
            rec_movies = self.recommend(user)  # 从recommend函数中获得一个推荐列表，拿这个算后来的recall和precision
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))


if __name__ == '__main__':
    start_time = time.time()
    rating_file = './data/ratings_100k.csv'
    itemCF = ItemBasedCF()
    itemCF.get_dataset(rating_file)
    itemCF.calc_movie_sim()
    itemCF.evaluate()
    end_time = time.time()
    print("Time elapse: %.2f s\n" % (end_time - start_time))
