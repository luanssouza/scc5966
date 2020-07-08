import pandas as pd
import numpy as np
from scipy.spatial import distance
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import time

start_alg = time.time()

train = pd.read_csv('dataset/train_data.csv', header=None,  skiprows=[0], usecols=[0,1,2])
test = pd.read_csv('dataset/test_data.csv')

mv = pd.read_csv('dataset/movie_reviews.csv')

train_movies = pd.read_csv('dataset/movies_data.csv')

def fbc_knn(train, features, k = 5):
    ratings = train.pivot(index=1, columns=0, values=2)
    ratings.fillna(0.0, inplace=True)

    sim = cosine_similarity(csr_matrix(features))
    sim = pd.DataFrame(data=sim, index=mv["movie_id"].unique())
    sim = pd.DataFrame(data=sim, index=train_movies["movie_id"].unique())

    sim.fillna(0.0, inplace=True)

    return { "sim": sim, "K": k, "ratings": ratings }

def predict(model, user, item, k = 5):
    sim = model["sim"]
    ratings = model["ratings"]
    if item not in sim or user not in ratings:
        return 0
    sim_items = sim[item].sort_values(ascending=False).index
    rated_items = ratings[user][ratings[user] > 0].index
    sim_k = np.intersect1d(sim_items, rated_items)
    top_k = []
    for x in sim_items:
        if k <= -1:
            break
        if x in sim_k:
            top_k.append(x)
            k-=1
    top_k = sim_k
    sumSim = 0.0
    sumWeight = 0.0
    for j in sim_k:
        sumSim += sim[item][j]
        sumWeight += sim[item][j] * ratings[user][j]
    if sumSim == 0.0:
        return 0

    return sumWeight/sumSim

def results(model, test):
    return [predict(model, t[1], t[2]) for t in test]

# Treinando
start_time = time.time()
v = TfidfVectorizer()
features = v.fit_transform(mv.groupby("movie_id")["text"].apply(lambda x: (x + " ").sum()))
fbc_knn = fbc_knn(train, features)
print("Tempo de treinamento em segundos: ", time.time() - start_time)

# Predizendo
start_time = time.time()
results = results(fbc_knn, test.values)
print("Tempo de predicao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/knn_fbc_naoestruturado_results.csv', encoding='utf-8', index=False)


print("Tempo de execucao em segundos: ", time.time() - start_alg)