from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

class ThresholdPCA(BaseEstimator,TransformerMixin):
  def __init__(self,threshold):
    self.threshold = threshold
    self.pca = TruncatedSVD(50)
    self.ncomponents = 0
  
  #faz o fit do PCA e procura pelo número mínimo de dimensões que explica a porcentagem "threshold" da variância.
  # se por algum motivo isso não der certo, temos um novo vetor de features com 50 dimensões e o método deixa registrado qual a porcentagem de variância explicada
  def fit(self,X,y=None):
    pca = self.pca.fit(X)
    #cumulative_explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    cumulative = 0
    for variance in cumulative_explained_variance:
      cumulative += variance
      self.ncomponents += 1
      if variance > self.threshold:
        break
    print("dimensão original:", X.shape)
    print("nova dimensão das features: ",self.ncomponents)
    #print("porcentagem explicada: ",cumulative)
    print("porcentagem explicada: ",variance)
    return self


  def transform(self,X,y=None):
    ret = self.pca.transform(X)
    return ret[:,:self.ncomponents]
