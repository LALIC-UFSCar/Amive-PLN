import json,spacy,pandas,pickle

sp = spacy.load("pt_core_news_sm")

with open("/home/augusto/Documentos/IC/pt-adapt-semi-supervised/Social-media-Depression-Detector/traduzido.json",'rb') as f:
    r = f.read()
termos = json.loads(r)

max = 0
sintomas = ["signal_1","signal_2","signal_3","signal_4","signal_5","signal_6","signal_7","signal_8","signal_9","signal_10"]
for sintoma in sintomas:
    for termo in termos[sintoma]:
        aux = len(termo.split("_")) - 1
        if max < aux:
            max = aux
#número máximo de n_gramas é 6.
print(max)
#um dataframe por sintoma
dataframes = [pandas.DataFrame() for i in range(10)]

##PARA POSTAGENS DO FACEBOOK:
#with open("/home/augusto/Documentos/IC/universidades.json" , "rb") as f:
    #p = f.read()
#posts = [json.loads(line)['Message']for line in p.splitlines()]

#PARA POSTAGENS DO CORPUS DO IVANDRÉ:
data = pandas.read_csv("/home/augusto/Documentos/IC/experimentos ivandré/lrec2020.csv",sep=";")
with open("/home/augusto/Documentos/IC/experimentos ivandré/replicar/wemb.pkl","rb") as f:
    embeddings = pickle.load(f)
#posts = data['Text'].to_numpy()

##PARA POSTAGENS DO REDDIT:
#data = pandas.read_csv("/home/augusto/Documentos/IC/experimentos reddit/reddit.csv")
#data = data.dropna(subset=['selftext'])
##posts = data['selftext'].to_numpy()


symptom_embeddings = [[] for i in range(10)]
print("tamanho do array de embeddings: ",len(embeddings))
print("tamanho do dataframe do corpus: ",data.shape)
#posts = ["testando","ausência de prazer"]
for index,post in data.iterrows():
    #esses provavelmente não são os nomes
    unigrams = []
    bigrams  = []
    trigrams = []
    quadgrams = []
    pentagrams = []
    hexagrams = []

    tokens = sp(post['Text'])
    for token in tokens:
        unigrams.append(token.text)

    for i, token in enumerate(tokens[:-1]):
        bigrams.append(token.text+"_"+tokens[i+1].text)

    for i, token in enumerate(tokens[:-2]):
        trigrams.append(token.text+"_"+tokens[i+1].text+"_"+tokens[i+2].text)

    for i, token in enumerate(tokens[:-3]):
        quadgrams.append(token.text+"_"+tokens[i+1].text+"_"+tokens[i+2].text+"_"+tokens[i+3].text)

    for i, token in enumerate(tokens[:-4]):
        pentagrams.append(token.text+"_"+tokens[i+1].text+"_"+tokens[i+2].text+"_"+tokens[i+3].text+tokens[i+4].text)

    for i, token in enumerate(tokens[:-5]):
        hexagrams.append(token.text+"_"+tokens[i+1].text +"_"+tokens[i+2].text+"_"+tokens[i+3].text+"_" + tokens[i+4].text + "_" + tokens[i+5].text)
    unigrams = set(unigrams)
    bigrams = set(bigrams)
    trigrams = set(trigrams)
    quadgrams = set(quadgrams)
    pentagrams = set(pentagrams)
    hexagrams = set(hexagrams)

    for i,sintoma in enumerate(sintomas):
        terms = set(termos[sintoma])
        #if not unigrams.isdisjoint(terms) or not bigrams.isdisjoint(terms) or not trigrams.isdisjoint(terms) or not quadgrams.isdisjoint(terms) or not pentagrams.isdisjoint(terms) or not hexagrams.isdisjoint(terms):
        post_terms  = unigrams.intersection(terms)  
        post_terms.update(bigrams.intersection(terms))
        post_terms.update(trigrams.intersection(terms))
        post_terms.update(quadgrams.intersection(terms))
        post_terms.update(pentagrams.intersection(terms))
        post_terms.update(hexagrams.intersection(terms))
        if post_terms:
            symptom_embeddings[i].append(embeddings[index])


for i,emb in enumerate(symptom_embeddings):
    filename = "sintomas_ivandré/sintoma_embeddings/" + str(i) +".pkl"
    series = pandas.Series(emb)
    with open(filename,"wb") as f:
        pickle.dump(series,f)