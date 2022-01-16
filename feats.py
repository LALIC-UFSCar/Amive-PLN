import re,sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from scipy.sparse import  csr, hstack, csr_matrix


#from enelvo import normaliser
import spacy

from sklearn.base import BaseEstimator, TransformerMixin

from sentiment import getSentiment
from liwc import Liwc
from gensim.utils import tokenize


#disponível em http://143.107.183.175:21380/portlex/index.php/pt/projetos/liwc
LIWC_PATH = "resources/LIWC2007_Portugues_win.dic.txt"

class FeatureExtract(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.liwc = Liwc(LIWC_PATH)
        #self.norm = normaliser.Normaliser()
        self.sp  = spacy.load("pt_core_news_sm") 
        self.tfidf = TfidfVectorizer()
        self.lda = LatentDirichletAllocation(random_state=10)
        self.count = CountVectorizer()

    def normalise_data(self,posts):
        ret = []
        i = 0
        for post in posts:
            n = self.norm.normalise(post).replace("number","" ).replace("hash","").replace("tag","").replace("link","")
            ret.append(n)
            if i%10000 == 0:
                print("normalizados ",i," de ",len(posts))
                i+= 1
        return ret


    def fit(self,X,y=None):
      
       self.tfidf = self.tfidf.fit(X['texto'])
       return self 

    def transform(self,X,y=None):
        ret = self.newExtractFeatures(X)
        return ret

    #categorias de related revisadas. Removendo: 
    # negemo(estudo aponta correlação bem fraca com sentimento)
    # feel (meio vago)
    # social (idem)
    # health (pensando melhor, não acho que isso vai capturar sintomas somáticos)
    def newExtractFeatures(self,posts,related= ["122","128","130","150","354","358","360","123"]  
):
        posts = posts['texto']
      
        frequencies = self.tfidf.transform(posts)
        print("frequencies shape: ",frequencies.shape)
        frequencies = csr_matrix(frequencies)

        #TODO: fazer disso um regex também.
        exclude = ["",".",":"," .","\"","\'"," "]
        featureVector = polarity_and_liwc(posts,related,self.liwc)
        print(featureVector.shape)

        return hstack([frequencies,featureVector])
 


def polarity_and_liwc(posts,related,liwc):
  featureVector = np.zeros((posts.shape[0],len(related) + 1),dtype=np.float32)
  l = len(posts)
  a = liwc.checkDoc
  for i,doc in enumerate(posts):     
            if(i%1000 == 0):
                print(i," de ",l)
            feats = np.zeros(len(related)+1)
            b = tokenize(doc)
            feats[:-1] = a(b,related)
            sent_split = splitSentences(doc)
            #mudando a forma como o cálculo é feito para uniformizar:
            #polaridade do tweet é a polaridade majoritária das sentenças
            positive = 0
            sentiments = getSentiment(sent_split)
            positive += sum(sentiments)
            negative = len(sentiments) - positive
            if len(sent_split) != 0:
              if negative > positive:
                feats[:-1] = -1
              else:
                feats[:-1] = 1
            else:
              feats[:-1] = 0
            featureVector[i] = feats

  featureVector = csr_matrix(featureVector)
  return featureVector


def splitSentences(doc):
  sentences = re.split(r'[!?.;][!?.;\s+"]*',doc)
  return sentences