import pandas as pd
import numpy as np
import math

import time

start_alg = time.time()

train = pd.read_csv('dataset/train_data.csv')
test = pd.read_csv('dataset/test_data.csv')

def knn(train):
    ratings = train.pivot(index='movie_id', columns='user_id', values='rating')
    sim = ratings.corr(method='pearson')
    return { "sim": sim, "ratings": ratings}

def predict(model, u, i, k = 5):
    sim = model["sim"]
    ratings = model["ratings"]
    ratings_T = ratings.T
    if u not in sim or i not in ratings_T:
        return 0
    sim_users = sim[u][~sim[u].isna()].sort_values(ascending=False).index
    rated_users = ratings_T[i][ratings_T[i] > 0].index
    sim_k = np.intersect1d(sim_users, rated_users)
    top_k = [x for x in sim_users if x in sim_k][:k]
    sum_sim = 0
    dem = 0
    mean_u = ratings[u].mean(skipna = True)
    for v in top_k:
        dem += sim[u][v] * (ratings[v][i] - ratings[v].mean(skipna = True))
        sum_sim += sim[u][v]
    if sum_sim == 0:
        return 0
    return mean_u + (dem/sum_sim)

def results(model, test):
    return [predict(model, t[1], t[2]) for t in test]

# Treinando
start_time = time.time()
knn = knn(train)
print("Tempo de treinamento em segundos: ", time.time() - start_time)

# Predizendo
start_time = time.time()
results = results(knn, test.values)
print("Tempo de predicao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/knn_users_results.csv', encoding='utf-8', index=False)


print("Tempo de execucao em segundos: ", time.time() - start_alg)