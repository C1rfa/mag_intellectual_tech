from nltk.parse import CoreNLPParser
from nltk.corpus import wordnet
from nltk import  word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import pandas as pd
import string
import re

def lemmatize(input_list):
    sentence = []
    for item in input_list:
        tag = get_wordnet_pos(item[1])
        if tag != None:
            sentence.append(WordNetLemmatizer().lemmatize(item[0],tag))
        else:
            sentence.append(item[0])
    return " ".join(sentence)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def position_tag(input_str):
    return pos_tag(tokenize(input_str))

def tokenize(input_str):
    return word_tokenize(input_str)

def text_to_lower(input_str):
    return input_str.lower()

def remove_numbers(input_str):
    return re.sub(r'\d+', '', input_str)

def remove_punctuation(input_str):
    return input_str.translate(str.maketrans("","", string.punctuation))

def remove_whitespaces(input_str):
    return " ".join(input_str.split())

def remove_url(input_str):
    return re.sub(r'https\+S|www\.\+S|http?', '', input_str)

def remove_stop_words(input_list):
    sentence = []
    for word in input_list:
        if word not in ENGLISH_STOP_WORDS and len(word) > 2 and word and word != len(word) * word[0]:
            sentence.append(word) 
    return " ".join(sentence)
    
    

def clear_text(input_str):
    return remove_whitespaces(remove_url(remove_punctuation(remove_numbers(text_to_lower(input_str)))))

# def remove_human_names_and_numbers(input_str):
#     tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
#     tokens = tokenize(input_str.title())
#     tags = tagger.tag(tokens)
#     sentence = []
#     for i in tags:
#         if i[1] != 'NUMBER' and i[1] != "PERSON" and i[1] != "DATE" and i[1] != "DURATION":
#             sentence.append(i[0])
    
#     return " ".join(sentence)
    
    