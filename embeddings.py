
import numpy as np
import spacy

from sklearn.base import BaseEstimator, TransformerMixin
class Embeddings(BaseEstimator, TransformerMixin):
  def __init__(self,model,max_length=0):
    self.model = model
    self.sp  = spacy.load("pt_core_news_sm") 
    self.max_length = max_length
  def fit(self,documents,y=None):
    #Talvez eu tente fazer padding depois se a média não der certo.
    #for doc in documents:
    #  tokens = sp(doc)
    #  if len(tokens) > self.max_length:
    #    self.max_length = len(tokens)
    return self
  
  def transform(self,documents,y=None):
    average_embedding = np.zeros(100)
    doc_vectors = []
    for doc in documents:
      tokens = self.sp(doc)
      for token in tokens:
        #se a palavra não está no vocabulário não tem muito o que fazer
        try:
          average_embedding += self.model[token.text]
        except:
          pass
      doc_vectors.append(average_embedding/len(tokens))
    print("número de exemplos: ",len(doc_vectors))
    return doc_vectors

