import pandas as pd
import numpy as np
import math

# Importando .csv
df = pd.read_csv('dataset/train_data.csv')

# Criando Matrix User X Movie
df_u_m = df.pivot(index='user_id', columns='movie_id', values='rating')

# Criando Matrix Movie X User
df_m_u = df.pivot(index='movie_id', columns='user_id', values='rating')

# Calculando similaridade de usuários com correlação de Pearson
def sim(u, v):
  # Preciso considerar apenas os itens avaliados por ambos os usuários
  sum_iuv = 0
  sum_sqrt_iu = 0
  sum_sqrt_iv = 0
  vies_u = u.mean(skipna = True)
  vies_v = v.mean(skipna = True)
  u = u.tolist()
  v = v.tolist()
  for i in range(1, len(u)):
    if(not pd.isnull(u[i]) and not pd.isnull(v[i])):
      riu = u[i] - vies_u
      riv = v[i] - vies_v
      sum_iuv += (riu*riv)
      sum_sqrt_iu += riu**2
      sum_sqrt_iv += riv**2
  return sum_iuv / math.sqrt(sum_sqrt_iu*sum_sqrt_iv)

# O próprio pandas implementa isso
# df_u_m.corr(method='pearson')
# print(df_m_u.corr(method='pearson'))
# print(df_m_u)

sim_u_v = df_m_u.corr(method='pearson')

def pred(u, i):
  top_k = sim_u_v.nlargest(3, u).index
  sum_sim = 0
  dem = 0
  mean_u = df_m_u[u].mean(skipna = True)
  for v in top_k:
    if (pd.isnull(df_m_u[v][i])):
        continue
    dem += sim_u_v[u][v] * (df_m_u[v][i] - df_m_u[v].mean(skipna = True))
    sum_sim += sim_u_v[u][v]
  return mean_u + (dem/sum_sim)

print(pred(1, 683))