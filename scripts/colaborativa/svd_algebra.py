import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import math

import time

start_alg = time.time()

train = pd.read_csv('dataset/train_data.csv')
train_movies = pd.read_csv('dataset/movies_data.csv')
train_users = pd.read_csv('dataset/users_data.csv')

test = pd.read_csv('dataset/test_data.csv')

def svd_algebra(train):
    global_mean = train["rating"].mean()
    users = pd.Series(range(1,train["user_id"].max()+1), name="user_id")
    movies = pd.Series(range(1,train["movie_id"].max()+1), name="movie_id")
    df_u_m = train.pivot(index='user_id', columns='movie_id', values='rating')
    df_u_m = pd.DataFrame(df_u_m, index=users, columns=movies)
    df_u_m.fillna(value=0, inplace=True)
    U, s, Vh = np.linalg.svd(df_u_m.values)
    return { "u": U, "s": s, "v": Vh }

def predict(model, u,i, k = 5):
    p2 = model["u"][u, :k]
    q2 = model["v"][i, :k]
    s2 = model["s"][:k]
    return p2.dot(np.diag(s2)).dot(q2)

def results(model, test):
    return [predict(model, t[1]-1, t[2]-1) for t in test]


# Treinando
start_time = time.time()
svd_algebra = svd_algebra(train)
print("Tempo de treinamento em segundos: ", time.time() - start_time)

# Predizendo
start_time = time.time()
results = results(svd_algebra, test.values)
print("Tempo de predicao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/svd_algebra_results.csv', encoding='utf-8', index=False)


print("Tempo de execucao em segundos: ", time.time() - start_alg)