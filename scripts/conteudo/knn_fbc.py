import pandas as pd
import numpy as np
from scipy.spatial import distance
import math

import time

train = pd.read_csv('dataset/train_data.csv', header=None,  skiprows=[0], usecols=[0,1,2])
test = pd.read_csv('dataset/test_data.csv')

m = pd.read_csv('dataset/movies_data.csv')
genres = pd.get_dummies(m.set_index(['movie_id']).genres.str.split('|', expand=True).stack(dropna=False)).sum(level=0)

def fbc_knn(train, features, K = 4):
    ratings = train.pivot(index=1, columns=0, values=2)
    ratings.fillna(0.0, inplace=True)
    sim = genres.T.corr(method=distance.jaccard)
    sim.fillna(0.0, inplace=True)

    return { "sim": sim, "K": K, "ratings": ratings }

def predict(model, user, item, k = 5):
    sim = model["sim"]
    ratings = model["ratings"]
    if item not in sim or user not in ratings:
        return 0
    sim_items = sim[item].sort_values(ascending=False).index
    rated_items = ratings[user][ratings[user] > 0].index
    sim_k = np.intersect1d(sim_items, rated_items)
    sim_k = [x for x in sim_items if x in sim_k][:k]
    sumSim = 0.0
    sumWeight = 0.0
    for j in sim_k:
        sumSim += sim[item][j]
        sumWeight += sim[item][j] * ratings[user][j]
    if sumSim == 0.0:
        return 0

    return sumWeight/sumSim

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

features = genres.values
features = np.hstack((features,np.ones((len(features),1))))

fbc_knn = fbc_knn(train, features)

results = results(fbc_knn, test.values)

# Finalizando contagem
print("Tempo de execucao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/knn_fbc_results.csv', encoding='utf-8', index=False)


