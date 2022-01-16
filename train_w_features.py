from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2

from gensim.models import KeyedVectors

import json,pickle,sys
import pandas as pd
import numpy as np

#comentar até lembrar de como instalar o enelvo corretamente
from feats import FeatureExtract
from embeddings import Embeddings
from tpca import ThresholdPCA

folder = sys.argv[1]
dest = sys.argv[2]

rstate = 10
#model = KeyedVectors.load_word2vec_format("/home/augusto/Documentos/IC/STIL 2021/cbow_s100.txt")

#originalmente : [2,5,7]
for i in range(9):

        data = pd.read_csv(f"{folder}/Cópia de sintoma_{i}_normalized.csv")[:1000]
        #tratando alguns casos em que o enelvo falhou (geralmente textos muito curtos com emojis)
        data = data.fillna({'texto':" "})

        #pelo visto tem que dar uma embaralhada, visto que as entradas estão em ordem de autor
        data = data.sample(frac=1,random_state=10).reset_index()

        print(data.shape)
        
        #balanceando o córpus
        X_pos = data.loc[data['Class'] == "yes"]
        X_neg = data.loc[data['Class'] == "no"]
        if len(X_pos) > len(X_neg):
            X_pos = X_pos.sample(len(X_neg),random_state=rstate)
        else:
            X_neg = X_neg.sample(len(X_pos),random_state=rstate)


        data = pd.concat([X_pos,X_neg])
        print(data.shape)

        y = LabelEncoder().fit_transform(data["Class"])
        print("shape de Y")
        print(y.shape)
        
        #X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.1,random_state=10)
        #X_train, X_test, y_train, y_test = train_test_split(data['texto'],y,test_size=0.1,random_state=10) 

        #substituir SVC() por modelo que desejar
        #É possível treinar embeddings a partir destes subconjuntos de tweets, substitua FeatureExtract() por Embeddings(), mas preferiu-se treinar com o corpus completo
        clf = make_pipeline(FeatureExtract(),ThresholdPCA(threshold=0.9),SVC())
        
        scoring = ['precision','recall','f1','accuracy','f1_micro','precision_micro']
        scores = cross_validate(clf,data,y,scoring=scoring,cv=5)
        

        with open (f"{dest}/sintoma_{i}_features.txt","w") as f:
            for key in scores.keys():
                scores[key] = scores[key].tolist() 
            json.dump(scores,f)

#trecho comentado salva o extrator, tpca e classificador em formato pickle para classificar o córpus inteiro e treinar o classificador de PPD
#(não consegui achar jeito de salvar o pipeline como pickle)
'''
        X_train, X_test, Y_train, Y_test = train_test_split(data,y,test_size=0.1,random_state=10)
        extract = FeatureExtract() 
        X_train = extract.fit_transform(X_train)
        with open(f"final/extract_{i}.pkl","wb") as f:
            pickle.dump(extract,f)
        tpca = ThresholdPCA(threshold=0.9)
        X_train = tpca.fit_transform(X_train)
        with open(f"final/tpca_{i}.pkl","wb") as f:
            pickle.dump(tpca,f)
        clf = MLPClassifier(max_iter=300)
        clf.fit(X_train,Y_train) 
        with open(f"final/sintoma_{i}.pkl","wb") as f:
            pickle.dump(clf,f)
        print(classification_report(clf.predict(tpca.transform(extract.transform(X_test))),Y_test))
'''