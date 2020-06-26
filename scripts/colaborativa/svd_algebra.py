import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import math

import time

train = pd.read_csv('dataset/train_data.csv')
train_movies = pd.read_csv('dataset/movies_data.csv')
train_users = pd.read_csv('dataset/users_data.csv')

test = pd.read_csv('dataset/test_data.csv')

def svd_algebra(train, movies, users):
    global_mean = train["rating"].mean()
    df_u_m = train.pivot(index='user_id', columns='movie_id', values='rating')
    df_u_m = pd.DataFrame(df_u_m, index=users["user_id"], columns=movies["movie_id"])
    df_u_m.fillna(value=global_mean, inplace=True)
    # df_u_m.fillna(value=0, inplace=True)
    U, s, Vh = np.linalg.svd(df_u_m.values)
    return { "u": U, "s": s, "v": Vh }

def predict(model, u,i, k = 5):
    p2 = model["u"][u, :k]
    q2 = model["v"][i, :k]
    s2 = model["s"][:k]
    return p2.dot(np.diag(s2)).dot(q2)

def rmse(test):
    sum_err = 0
    for t in test:
        u = t[0]-1
        i = t[1]-1
        r_ui = t[2]
        pred = predict(u, i)
        error = (r_ui - pred)**2
        sum_err += error
    return math.sqrt(sum_err/len(test))

def results(model, test):
    return [predict(model, t[1]-1, t[2]-1) for t in test]

# Iniciando contagem
start_time = time.time()

svd_algebra = svd_algebra(train, train_movies, train_users)

results = results(svd_algebra, test.values)

# Finalizando contagem
print("Tempo de execucao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/svd_algebra_results.csv', encoding='utf-8', index=False)

