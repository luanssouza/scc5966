import pandas as pd
import numpy as np
import math

# Importando .csv
df = pd.read_csv('dataset/train_data.csv')

# Criando Matrix User X Movie
df_u_m = df.pivot(index='user_id', columns='movie_id', values='rating')

# Criando Matrix Movie X User
df_m_u = df.pivot(index='movie_id', columns='user_id', values='rating')

# Trabalhando com método baseline
mean = df_u_m.stack().mean(skipna = True)

def item_bias(col):
    return col.mean(skipna = True) - mean
    
def user_bias(col):
    return col.mean(skipna = True) - mean

print(mean + user_bias(df_m_u[1]) + item_bias(df_u_m[1]))