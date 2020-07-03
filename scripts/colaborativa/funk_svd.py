import pandas as pd
import numpy as np
from scipy.spatial import distance
import math

import time

start_alg = time.time()

train = pd.read_csv('dataset/train_data.csv', header=None,  skiprows=[0], usecols=[0,1,2])
test = pd.read_csv('dataset/test_data.csv')

m = pd.read_csv('dataset/movies_data.csv')
genres = pd.get_dummies(m.set_index(['movie_id']).genres.str.split('|', expand=True).stack(dropna=False)).sum(level=0)

def funkSVD(train, k = 5, lr = 0.05, reg = 0.02, miter = 10):
    global_mean = train[2].mean(skipna = True)
    nusers = train[0].max()+1
    nitems = train[1].max()+1
    bu = np.full(nusers, 0, dtype=float)
    bi = np.full(nusers, 0, dtype=float)
    P = np.random.normal(loc = 0, scale = 0.1, size=(nusers, k))
    Q = np.random.normal(loc = 0, scale = 0.1, size=(nitems, k))
    for f in range(k):
        for l in range(0, miter):
            for j in train.index:
                u = train[0][j]
                i = train[1][j]
                r_ui = train[2][j]
                pred = global_mean + bu[u] + bi[i] + np.dot(P[u, ], Q[i, ])
                e_ui = r_ui - pred
                bu[u] = bu[u] + lr * e_ui # - reg * bu[u]
                bi[i] = bi[i] + lr * e_ui # - reg * bi[i]
                temp_uf = P[u, f]
                P[u, f] = P[u, f] + lr * (e_ui * Q[i, f] - reg * P[u, f])
                Q[i, f] = Q[i, f] + lr * (e_ui * temp_uf - reg * Q[i, f])

    return { "global_mean": global_mean, "bu": bu, "bi": bi, "P": P, "Q": Q }


def predict(model, u, i):
    return model["global_mean"] + model["bu"][u] + model["bi"][i] + np.dot(model["P"][u], model["Q"][i])

def results(model, test):
    return [predict(model, t[1], t[2]) for t in test]

# Treinando
start_time = time.time()
funkSVD = funkSVD(train, k = 5, lr = 0.05, reg = 0.02, miter = 10)
print("Tempo de treinamento em segundos: ", time.time() - start_time)

# Predizendo
start_time = time.time()
results = results(funkSVD, test.values)
print("Tempo de predicao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/funk_svd_results.csv', encoding='utf-8', index=False)


print("Tempo de execucao em segundos: ", time.time() - start_alg)