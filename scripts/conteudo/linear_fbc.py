import pandas as pd
import numpy as np
import math

import time

start_alg = time.time()

train = pd.read_csv('dataset/train_data.csv', header=None,  skiprows=[0], usecols=[0,1,2])
test = pd.read_csv('dataset/test_data.csv')

m = pd.read_csv('dataset/movies_data.csv')
genres = pd.get_dummies(m.set_index(['movie_id']).genres.str.split('|', expand=True).stack(dropna=False)).sum(level=0)


def fbc_linear(train, test, features, lr = 0.05, reg = 0.002, miter = 10):
    nusers = np.append(train[:,0], test[:,0]).max()
    nitems = np.append(train[:,1], test[:,1]).max()
    nfeatures = len(features[1])-1
    # features = np.hstack((features,np.ones((len(features),1))))
    profiles = np.random.normal(loc = 0, scale = 0.1, size=(nusers, nfeatures+1))
    error = list()
    for l in range(0, miter):
        sq_error = 0
        for j in range(0, len(train)):
            u = train[j, 0]-1
            i = train[j, 1]-1
            r_ui = train[j, 2]-1
            e_ui = np.dot(profiles[u, ], features[i, ]) - r_ui
            sq_error += e_ui**2
            for k in range(nfeatures-1):
                profiles[u, k] = profiles[u, k] - lr * (e_ui * features[i, k] + reg * profiles[u, k])
            k = nfeatures
            profiles[u, k] = profiles[u, k] - lr * (e_ui * features[i, k])
        error.append(math.sqrt(sq_error/len(train)))
    return { "profiles": profiles, "error": error }


def predict(model, user, item, features):
    return np.dot(model["profiles"][user-1, ], features[item-1, ])

def results(model, test, features):
    return [predict(model, t[1], t[2], features) for t in test]

features = genres.values
features = np.hstack((features,np.ones((len(features),1))))

# Treinando
start_time = time.time()
fbc_linear = fbc_linear(train.values[:482205], train.values[482205:], features)
print("Tempo de treinamento em segundos: ", time.time() - start_time)

# Predizendo
start_time = time.time()
results = results(fbc_linear, test.values, features)
print("Tempo de predicao em segundos: ", time.time() - start_time)

results = pd.DataFrame({ 'rating': results })
results.insert(0, 'id', results.index)
results.to_csv('results/linear_fbc_results.csv', encoding='utf-8', index=False)


print("Tempo de execucao em segundos: ", time.time() - start_alg)