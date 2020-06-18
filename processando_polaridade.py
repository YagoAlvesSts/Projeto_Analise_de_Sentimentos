import gensim
import numpy as np
import os
import pickle
import fnmatch
from enelvo import normaliser

#realiza tratamento de texto
def pre_processing_text(text, use_normalizer=False):

    if use_normalizer:
        norm = normaliser.Normaliser()
        text = norm.normalise(text)

    text = text.lower()

    input_chars = ["\n", ".", "!", "?", "ç", " / ", " - ", "|", "ã", "õ", "á", "é", "í", "ó", "ú", "â", "ê", "î", "ô", "û", "à", "è", "ì", "ò", "ù"]
    output_chars = [" . ", " . ", " . ", " . ", "c", "/", "-", "", "a", "o", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u"]

    for i in range(len(input_chars)):
        text = text.replace(input_chars[i], output_chars[i])  

    text.strip()

    return text



all_reviews_n = []
all_reviews_p = []
all_reviews_t = []

#busca reviews negativos (avaliados de 0 a 2) e atribui polaridade negativa
for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/negativos"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review_n = f.read()
        review_n = [pre_processing_text(review_n, use_normalizer=True), '-1']
        all_reviews_n.append(review_n)

#busca reviews positivos (avaliados de 3 a 5) e atribui polaridade positiva

for dirpath, _, files in os.walk("./Corpus Buscape/treinamento/positivos"):
    for filename in fnmatch.filter(files, '*.txt'):
        f = open(os.path.join(dirpath, filename), "r", encoding="utf8")
        review_p = f.read()
        review_p = [pre_processing_text(review_p, use_normalizer=True), '1']
        all_reviews_p.append(review_p)


#concatena reviews e adiciona em uma só lista
all_reviews_t = all_reviews_n + all_reviews_p
print(all_reviews_t)

#escreve reviews com polaridades
with open("Processed_Reviews_polarity.p", "wb") as f:
    pickle.dump(all_reviews_t, f)

