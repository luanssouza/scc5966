import pandas as pd
import numpy as np
import math

import time

train = pd.read_csv('dataset/train_data.csv', header=None,  skiprows=[0], usecols=[0,1,2])
test = pd.read_csv('dataset/test_data.csv')

def svdopt(train, k, lr = 0.05, reg = 0.02,miter = 10):
    global_mean = train[2].mean(skipna = True)
    nusers = train[0].max()+1
    nitems = train[1].max()+1
    bu = np.full(nusers, 0, dtype=float)
    bi = np.full(nusers, 0, dtype=float)
    P = np.random.normal(loc = 0, scale = 0.1, size=(nusers, k))
    Q = np.random.normal(loc = 0, scale = 0.1, size=(nitems, k))
    error = list()
    for l in range(0, miter):
        sq_error = 0
        for j in range(0, len(train)):
            u = train[0][j]
            i = train[1][j]
            r_ui = train[2][j]
            pred = global_mean + bu[u] + bi[i] + np.dot(P[u, ], Q[i, ])
            e_ui = r_ui - pred
            sq_error += e_ui**2
            bu[u] += lr * e_ui # - reg * bu[u]
            bi[i] += lr * e_ui # - reg * bi[i]
            for f in range(k):
                temp_uf = P[u, f]
                P[u, f] = P[u, f] + lr * (e_ui * Q[i, f] - reg * P[u, f])
                Q[i, f] = Q[i, f] + lr * (e_ui * temp_uf - reg * Q[i, f])
        error.append(math.sqrt(sq_error/len(train)))

    return { "global_mean": global_mean, "bu": bu, "bi": bi, "P": P, "Q": Q, "error": error }

def predict(model, u, i):
    return model["global_mean"] + model["bu"][u] + model["bi"][i] + np.dot(model["P"][u], model["Q"][i])


def rmse(model, test):
    sum_err = 0
    for t in test:
        u = t[0]
        i = t[1]
        r_ui = t[2]
        pred = predict(model, u, i)
        error = (r_ui - pred)**2
        sum_err += error
    return math.sqrt(sum_err/len(test))

def results(model, test):
    return [predict(model, t[1], t[2]) for t in test]

# Iniciando contagem
start_time = time.time()

svdopt = svdopt(train, 4)

results = results(svdopt, test.values)

# Finalizando contagem
print("Tempo de execucao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/svd_opt_results.csv', encoding='utf-8', index=False)