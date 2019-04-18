import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

file_path = './data/ratings_2700w.csv'
# 2775 3444
raw_data = pd.read_csv(file_path, dtype={'userId': str, 'movieId': str})


# file_path = './data/ratings_100k.csv'
# # 100837
# raw_data = pd.read_csv(file_path, dtype={'userId': str, 'movieId': str})
#
# # 50k:
# rate = 50000 * 1.0 / 100837
# train_set, test_set = train_test_split(raw_data, test_size=rate)
# test_set.to_csv('./data/ratings_50k.csv', index=False)


# 200k:
rate = 200000 * 1.0 / 27753444
train_set, test_set = train_test_split(raw_data, test_size=rate)
test_set.to_csv('./data/ratings_200k.csv', index=False)

# # 500k:
# rate = 500000 * 1.0 / 27753444
# train_set, test_set = train_test_split(raw_data, test_size=rate)
# test_set.to_csv('./data/ratings_500k.csv', index=False)

# # 100w:
# rate = 1000000 * 1.0 / 27753444
# train_set, test_set = train_test_split(raw_data, test_size=rate)
# test_set.to_csv('./data/ratings_100w.csv', index=False)

# # 150w:
# rate = 1500000 * 1.0 / 27753444
# train_set, test_set = train_test_split(raw_data, test_size=rate)
# test_set.to_csv('./data/ratings_150w.csv', index=False)
