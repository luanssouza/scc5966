import pandas as pd
import numpy as np
import math

import time

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

    return { "bu" : bu, "bi": bi, "mean": mean }

def predict(model, u, i):
    return model["mean"] + model["bu"][u] + model["bi"][i]

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

baseline = baseline(train)

results = results(baseline, test.values)

# Finalizando contagem
print("Tempo de execucao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/baseline_results.csv', encoding='utf-8', index=False)


