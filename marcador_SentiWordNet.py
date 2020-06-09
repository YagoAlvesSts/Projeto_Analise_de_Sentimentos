import nltk
import os
import pickle
from nltk.corpus import sentiwordnet as swn
nltk.download('sentiwordnet')
nltk.download('wordnet')

pos_reviews_path = "rt_polarity_pos"
neg_reviews_path = "rt_polarity_neg"


print(list(swn.senti_synsets('good'))[3].neg_score())



"""

def sentiwordnet_classify(text):
        for i,review in enumerate(text):
                score_tot = 0
                score_tot_thr = 0
                class_tot = 0
                class_tot_thr = 0
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                (score, score_thr) = sentence_score(sentence)
                score_tot += score
                score_tot_thr += score_thr
                 
                #Trust the thresholded value more when classifying
                if score_tot_thr != 0:
                        clss = 'Positive' if score_tot_thr > 0 else 'Negative'
                        elif score_tot != 0:
                                clss = 'Positive' if score_tot > 0 else 'Negative'
                        else:
                                clss = 'Neutral'
                        return clss

def sentence_score(text, threshold = 0.75, wsd = word_sense_cdf):
        
        tagged_words = pos_tag(text)
 
        obj_score = 0 # object score 
        pos_score=0 # positive score
        neg_score=0 #negative score
        pos_score_thr=0
        neg_score_thr=0
 
        for word in tagged_words:
        #     print word
                if 'punct' not in word :
                    sense = wsd(word['word'], text, wordnet_pos_code(word['pos']))
                    if sense is not None:
                        sent = swn.senti_synset(sense.name())
                        if sent is not None and sent.obj_score() <> 1:
                            obj_score = obj_score + float(sent.obj_score())
                            pos_score = pos_score + float(sent.pos_score())
                            neg_score = neg_score + float(sent.neg_score())
                            if sent.obj_score() < threshold:
                                pos_score_thr = pos_score_thr + float(sent.pos_score())
                                neg_score_thr = neg_score_thr + float(sent.neg_score())
 
            return (pos_score - neg_score, pos_score_thr - neg_score_thr)


"""
text = []

with open(os.path.join("USO_GERAL.p"), "rb") as file:
        text = pickle.load(file)



