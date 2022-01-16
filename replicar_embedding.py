
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import sys,spacy,pickle

sp = spacy.load("pt_core_news_sm")

#path do csv com os tweets
folder = sys.argv[1]
data = pd.read_csv(folder,sep=";")
y = LabelEncoder().fit_transform(data['Class'])


#embeddings usando o word2vec skipgram (pelo jeito essas não eram pré-treinadas)

#docs= []
#i = 0
#for row in data.itertuples():
    #tokens = []
    #for token in sp(row.Text):
        #tokens.append(token.text.lower())
    #docs.append(tokens)
    #i += 1
    #if i%1000 == 0:
        #print(i)
#print("OK")
#with open("docs.pkl","wb") as f:
    #pickle.dump(docs,f)

#o trecho comentado acima salva este arquivo .pkl para salvar tempo em execuções subsequentes
with open("docs.pkl","rb") as f:
    docs = pickle.load(f)
embedmodel = Word2Vec(docs,sg=1)
tfidfvec = TfidfVectorizer()
tfidfmatrix = tfidfvec.fit_transform(data['Text'])
weight_dict = dict(zip(tfidfvec.get_feature_names(),tfidfvec.idf_))

def get_average_weighted_embedding(emb,w_dict,sp,doc):
    tokens = sp(doc)
    avg_w_embed = np.zeros(100)
    i = 0
    for token in tokens:
        try:
            cur_embed = np.array(emb[token.text.lower()])
            avg_w_embed += cur_embed * np.array(w_dict[token.text.lower()])
            i += 1
        except:
            pass
    if i > 0:
        return avg_w_embed/i
    else:
        return avg_w_embed

wemb = data.apply(lambda x: get_average_weighted_embedding(embedmodel,weight_dict,sp,x["Text"]),axis=1)
with open ("wemb.pkl","wb") as f:
    pickle.dump(wemb,f)
print("pronto!")
