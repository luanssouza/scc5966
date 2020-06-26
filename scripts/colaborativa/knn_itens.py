import pandas as pd
import numpy as np
import math

import time

train = pd.read_csv('dataset/train_data.csv')
test = pd.read_csv('dataset/test_data.csv')

def knn(train):
    ratings = train.pivot(index='user_id', columns='movie_id', values='rating')
    sim = ratings.corr(method='pearson')
    return { "sim": sim, "ratings": ratings}

def predict(model, u, i, k = 5):
    sim = model["sim"]
    ratings = model["ratings"]
    ratings_T = ratings.T
    if i not in sim or u not in ratings_T:
        return 0
    sim_items = sim[i][~sim[i].isna()].sort_values(ascending=False).index
    rated_items = ratings_T[u][ratings_T[u] > 0].index
    sim_k = np.intersect1d(sim_items, rated_items)
    top_k = [x for x in sim_items if x in sim_k][:k]
    sum_sim = 0
    dem = 0
    mean_u = ratings[i].mean(skipna = True)
    for v in top_k:
        dem += sim[i][v] * ratings[v][u] # - ratings[v].mean(skipna = True))
        sum_sim += sim[i][v]
    if sum_sim == 0:
        return 0
    return dem/sum_sim

def rmse(test):
    sum_err = 0
    for t in test:
        u = t[0]
        i = t[1]
        r_ui = t[2]
        pred = predict(u, i)
        error = (r_ui - pred)**2
        sum_err += error
    return math.sqrt(sum_err/len(test))

def results(model, test):
    return [predict(model, t[1], t[2]) for t in test]

# Iniciando contagem
start_time = time.time()

knn = knn(train)

results = results(knn, test.values)

# Finalizando contagem
print("Tempo de execucao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/knn_itens_results.csv', encoding='utf-8', index=False)

