from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from gensim.models import KeyedVectors

import json,pickle
import pandas as pd
import numpy as np
from embeddings import Embeddings
from tpca import ThresholdPCA
import sys

folder = sys.argv[1]
dest = sys.argv[2]
embeddings_folder = sys.argv[3]

#model = KeyedVectors.load_word2vec_format("/home/augusto/Documentos/IC/STIL 2021/cbow_s100.txt")

#originalmente : [2,5,7]
for i in [1]:
    data = pd.read_csv(f"{folder}/Cópia de sintoma_{i}_normalized.csv")
    data = data.dropna(subset=['texto'])

    #embeddings pré-processadas
    with open(f"sintomas_ivandré/sintoma_embeddings/{i}.pkl",'rb') as f:
        embedding = pickle.load(f)
    embedding = np.array(embedding) 
    embedding = np.vstack(embedding)
    embedding = pd.DataFrame(embedding)
    print("shape do df de embeddings: ", embedding.shape)
    print("shape do df do corpus: ",data.shape)
    data = pd.concat([data,embedding],axis=1)
    print("novo shape do df: ",data.shape)

    X_pos = data.loc[data['Class'] == "yes"]
    X_neg = data.loc[data['Class'] == "no"]
    if len(X_pos) > len(X_neg):
        X_pos = X_pos.sample(len(X_neg),random_state=10)
    else:
        X_neg = X_neg.sample(len(X_pos),random_state=10)

    print("tamanho X_pos: ",len(X_pos))
    print("tamanho X_neg: ",len(X_neg))

    data = pd.concat([X_pos,X_neg])

    data = data.sample(frac=1,random_state=10)
    X = data.iloc[:,-100:]
    print(X.shape)
    y = LabelEncoder().fit_transform(data["Class"])
    #X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.1,random_state=10)
       
    clf = SVC()
    scoring = ['precision','recall','f1','accuracy','f1_micro','precision_micro',]
    scores = cross_validate(clf,X,y,scoring=scoring,cv=5)
    with open (f"{dest}/sintoma_{i}_embeddings.txt","w") as f:
            for key in scores.keys():
                scores[key] = scores[key].tolist() 
            json.dump(scores,f)


    '''
    clf.fit(X_train,Y_train) 

    with open(f"final/sintoma_{i}.pkl","wb") as f:
        pickle.dump(clf,f)

    with open (f"modelos/svc/resultados/sintoma_{i}_embedding.txt","w") as f:
        f.write(f"resutados para sintoma_{i}:")
        f.write(classification_report(clf.predict(X_test),Y_test))
    '''

