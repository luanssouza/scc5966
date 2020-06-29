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
    
    bi = np.nan_to_num(bi)
    bu = np.nan_to_num(bu)
    model = { "bu" : bu, "bi": bi, "mean": mean }
    def predict(u, i):
        return model["mean"] + model["bu"][u] + model["bi"][i]

    model["pred"] = predict
    return model

def knn(train):
    ratings = train.pivot(index='movie_id', columns='user_id', values='rating')
    sim = ratings.corr(method='pearson')

    model = { "sim": sim, "ratings": ratings }
    def predict(u, i, k = 5):
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

    model["pred"] = predict
    return model

def hibrid(train):
    return { "baseline": baseline(train), "knn": knn(train) }

def predict(model, u, i, k = 5):
    return (0.5 * model["baseline"]["pred"](u, i)) + (0.5 * model["knn"]["pred"](u, i, k))

def results(model, test):
    return [predict(model, t[1], t[2]) for t in test]

# Iniciando contagem
start_time = time.time()

hibrid = hibrid(train)

results = results(hibrid, test.values)

# Finalizando contagem
print("Tempo de execucao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/hibrid_results.csv', encoding='utf-8', index=False)

