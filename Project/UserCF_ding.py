# coding:utf-8

import math
import random
import numpy as np
import time

NumOfUsers = 944
NumOfMovies = 1682
NumOfK = 20


def GetData(filename="./data/u.data"):
    """
    :param filename: File Path
    :return: 输出是List[(user_ID, movie_ID)]
    """
    data = []
    # user_set = set()
    # movie_set = set()
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("\t")
            data.append((int(line[0]), int(line[1])))
            # user_set.add(int(line[0]))
            # movie_set.add(int(line[1]))

        f.close()

    # print("user number: ", len(user_set))
    # print("movie number: ", len(movie_set))

    return data


def SplitData(data, M=9, rand_num=1, seed=3):
    """
    :param data:
    :param M:
    :param rand_num:
    :param seed:
    :return: _train,_test 是 Dict{key = user: value = set(movieID)}
    """
    _test = dict()
    _train = dict()
    random.seed(seed)
    # TODO: Use sklearn.train_and_test_split
    for user, item in data:
        if random.randint(0, M) == rand_num:
            if user not in _test.keys():
                _test[user] = set()
            _test[user].add(item)
        else:
            if user not in _train.keys():
                _train[user] = set()
            _train[user].add(item)
    return _train, _test


def Recall(train, test, N):
    hit = 0  # 预测准确数
    all = 0  # 所有行为总数
    W, related_users = UserIIFSimilarity(train)
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, train, N, W, related_users)
        for item in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)


def Precision(train, test, N):
    hit = 0
    precision_all = 0
    W, related_users = UserIIFSimilarity(train)
    for user in train.keys():
        if test.__contains__(user):
            tu = test[user]
            rank = GetRecommendation(user, train, N, W, related_users)
            for item in rank:
                if item in tu:
                    hit += 1
            precision_all += N
        else:
            print("User %d has problem" % user)
            continue
    return hit / (1.0 * precision_all)


def Recall_and_Precision(train, test, N):
    hit = 0
    recall_all = 0
    precision_all = 0

    W, related_users = UserIIFSimilarity(train)
    for user in train.keys():
        try:
            tu = test[user]
            rank = GetRecommendation(user, train, N, W, related_users)
            for item in rank:
                if item in tu:
                    hit += 1
            recall_all += len(tu)
            precision_all += N
        except KeyError:
            print("User %d has problem" % user)
            continue

    return hit / (1.0 * recall_all), hit / (1.0 * precision_all)


def Coverage(train, test, N):
    """
    :param N: N from topN
    :return:
    """
    recommend_items = set()
    all_items = set()
    W, related_users = UserIIFSimilarity(train)
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        rank = GetRecommendation(user, train, N, W, related_users)
        for item in rank:
            recommend_items.add(item)

    return len(recommend_items) / (1.0 * len(all_items))


def Popularity(train, test, N):
    item_popularity = dict()
    W, related_users = UserIIFSimilarity(train)
    for user, items in train.items():
        for item in items:
            item_popularity[item] = item_popularity.get(item, 0) + 1

    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user, train, N, W, related_users)
        for item in rank:
            if item != 0:
                ret += math.log(1 + item_popularity[item])
                n += 1
    return ret / (1.0 * n)


def CosineSimilarity(train):
    """
    计算余弦相似度，复杂度太高，程序中不用，主要用于理解公式原型
    :param train:
    :return: W 相似度矩阵
    """
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u == v:
                continue
            W[(u, v)] = len(train[u] & train[v])
            W[(u, v)] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
            W[(v, u)] = W[(u, v)]
    return W


def UserIIFSimilarity(train):
    """
    :param train:
    :return: 返回用户相似度矩阵W[u][v],
    """
    # Build inverse table for item_users 建立电影->用户倒排表
    item_user_inv_table = dict()
    for u, items in train.items():
        for i in items:
            if i not in item_user_inv_table:
                item_user_inv_table[i] = set()
            item_user_inv_table[i].add(u)
    print("Item_User inverse table completed\n")

    # Calculate co-rated items between users
    # C[u][v] 表示用户u和用户v之间共同喜欢的电影
    C = np.zeros([NumOfUsers, NumOfUsers], dtype=np.float16)
    # N[u]表示u评价的电影数量
    N = np.zeros([NumOfUsers], dtype=np.int32)
    # 表示u的相关用户，即共同电影不为0的用户
    user_related_users = dict()     # dict{key=userID : value = set(other userID)}

    # 对每个电影，把对应的C[u][v]加1,同时添加惩罚项
    for item, users in item_user_inv_table.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                if u not in user_related_users:
                    user_related_users[u] = set()
                user_related_users[u].add(v)
                # 这块的惩罚项对吗？？？
                C[u][v] += (1 / math.log(1 + len(users)))

    # Calculate final similarity matrix W 得到用户相似度矩阵
    W = np.zeros([NumOfUsers, NumOfUsers], dtype=np.float16)
    for u in range(1, NumOfUsers):
        if u in user_related_users:
            for v in user_related_users[u]:
                W[u][v] = C[u][v] / math.sqrt(N[u] * N[v])

    return W, user_related_users


def RecommendInterest(user, train, W, related_users):
    """
    :param user:
    :param train:
    :param W:
    :param related_users:
    :return:
    """
    rank = dict()
    for i in range(1, NumOfMovies + 1):
        rank[i] = 0
    k_users = dict()
    try:
        for v in related_users[user]:
            k_users[v] = W[user][v]
    except KeyError:
        print("User " + str(user) + " doesn't have any related users in train set")

    k_users = sorted(k_users.items(), key=lambda x: x[1], reverse=True)
    k_users = k_users[0:NumOfK]  # 取前k个用户

    for i in range(NumOfMovies + 1):
        for v, wuv in k_users:
            # 取出被user相似用户v产生行为的电影，同时user没有和这部电影产生行为
            if i in train[v] and i not in train[user]:
                rank[i] += wuv * 1

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)


def GetRecommendation(user, train, N, W, related_users):
    """
    :param user: 用户
    :param train: 训练集
    :param N: 推荐数量
    :param k: 从相似用户中取多少个
    :param W: 相似度矩阵
    :param related_users: recommend_interest 即用户对电影的感兴趣程度
    :return:
    """
    rank = RecommendInterest(user, train, W, related_users)
    recommend = dict()
    for i in range(N):
        recommend[rank[i][0]] = rank[i][1]
    return recommend


def Evaluate(train, test, N):
    recommends = dict()
    # W, related_users = UserIIFSimilarity(train)
    # for user in test:
    # for user in range(5):
    #     recommends[user] = GetRecommendation(user, train, N, W, related_users)

    recall, precision = Recall_and_Precision(train, test, N)
    coverage = Coverage(train, test, N)
    popularity = Popularity(train, test, N)
    # coverage = 0
    # popularity = 0
    return recall, precision, coverage, popularity


def test():
    # N = 20
    # k = 10
    data = GetData()
    print("data length: ", len(data))
    # data.sort()
    # for i in range(20):
    #     print(data[i])
    # print(data[-1])

    train_data, test_data = SplitData(data)
    # for k, v in train_data.items():
    #     print(k, v)
    # print(type(train_data))
    # print(type(train_data[1]))
    # v_sum = 0
    # tv_sum = 0
    # for k, v in train_data.items():
    #     for vv in v:
    #         v_sum += 1
    # for k, v in test_data.items():
    #     for vv in v:
    #         tv_sum += 1
    # print(v_sum)
    # print(tv_sum)
    print("train length: ", len(train_data))
    print("test length: ", len(test_data))
    # UserIIFSimilarity(train_data)
    del data

    N = 20

    recall, precision, coverage, popularity = Evaluate(train_data, test_data, N)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("Coverage: ", coverage)
    print("Popularity: ", popularity)


if __name__ == '__main__':
    start_time = time.time()
    test()
    end_time = time.time()
    print("Elapse : %.2f s\n" % (end_time - start_time))
