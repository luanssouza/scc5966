import pandas as pd
import numpy as np
import math

import time

start_alg = time.time()

train = pd.read_csv('dataset/train_data.csv')
test = pd.read_csv('dataset/test_data.csv')

def baseline(train):
    mean = train["rating"].mean(skipna = True)
    users = train.pivot(index='movie_id', columns='user_id', values='rating')
    items = train.pivot(index='user_id', columns='movie_id', values='rating')
    nusers = train.values[:, 0].max()+1
    nitems = train.values[:, 1].max()+1
    bu = np.full(nusers, np.nan, dtype=float)
    bi = np.full(nitems, np.nan, dtype=float)

    for i in items.columns.values:
        bi[i] = np.nanmean(items[i] - mean)

    aux = bi[~np.isnan(bi)]
    for u in users.columns.values:
        bu[u] = np.nanmean(users[u] - mean - aux)

    bi = np.nan_to_num(bi)
    bu = np.nan_to_num(bu)
    
    return { "bu" : bu, "bi": bi, "mean": mean }

def predict(model, u, i):
    return model["mean"] + model["bu"][u] + model["bi"][i]

def results(model, test):
    return [predict(model, t[1], t[2]) for t in test]

# Treinando
start_time = time.time()
baseline = baseline(train)
print("Tempo de treinamento em segundos: ", time.time() - start_time)

# Predizendo
start_time = time.time()
results = results(baseline, test.values)
print("Tempo de predicao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/baseline_results.csv', encoding='utf-8', index=False)


print("Tempo de execucao em segundos: ", time.time() - start_alg)