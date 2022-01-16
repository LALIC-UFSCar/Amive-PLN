import os
import numpy as np
from  numba import njit
#TODO: refatorar isso quando sobrar tempo pq tá feio demais
class Liwc:
    def __init__(self,filename):
        with open(filename,"r",encoding="windows-1252") as f:
            fstr = f.read()
        lines = fstr.split("\n")
        self.categories = {}
        #essas são as categorias
        for line in lines[1:65]:
            pair = line.split("\t")
            self.categories[pair[0]] = pair[1]
        self.words = {}
        for line in lines[67:]:
            labels = line.split("\t")
            for label in labels[1:]:
                if labels[0] not in self.words:
                    self.words[labels[0]] = []
                self.words[labels[0]].append(label)

        print("dicionários inicializados")

    def checkWord(self,word,l=[],use_all=False):
        '''checa se uma palavra pertence a um subconjunto de categorias do LIWC
        word(string): a palavra que se deseja verificar
        l(lista): lista de valores de categorias do LiWC

        exemplo: checar se a palavra tio faz parte de família:
            licw = Liwc(doc_path)
            licw.checkWord(tio,["122"])
        '''
        try:
            ret = []
            w_categories = self.words[word]
            if use_all:
                for cat in w_categories:
                    ret.append(int(cat))
            else:
                for cat in w_categories:
                    if cat in l:
                        ret.append(int(cat))
            return ret
            
        except KeyError:
            return []
    #contagem das palavras de cada categoria no documento.
    #doc deve estar na forma tokenizada.
    def checkDoc(self,doc,categories=[]):
        #mapeamento categoria-posição no vetor de retorno:
        categ_index = {}
        for categ in enumerate(categories):
            categ_index[int(categ[1])] = categ[0]

        counts = [0] * len(categories)
        for word in doc:
            categs = self.checkWord(word,categories)
            for categ in categs:
                if str(categ) in categories:
                    counts[categ_index[int(categ)]] += 1
        return counts
 

#parecido com o de cima, mas só retorna presença ao invés de contagem.
#possivelmente melhor para textos curtos.
    def checkPresenceDoc(self,doc,categories=[]):
        #mapeamento categoria-posição no vetor de retorno:
        categ_index = {}
        for i,categ in enumerate(categories):
            categ_index[int(categ)] = i

        presence = [False] * len(categories)
        for word in doc:
            print(word)
            categs = self.checkWord(word,categories)
            for categ in categs:
                if str(categ) in categories:
                    presence[categ_index[int(categ)]] = True
        return presence


    def checkEncoding(self,cat):
        try:
            return self.categories[cat]
        except:
            return ""

