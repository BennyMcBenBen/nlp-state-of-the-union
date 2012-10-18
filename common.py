'''
Created on May 5, 2010

@author: Ben
'''
from __future__ import division
import math
import re
from heapq import nlargest
from nltk import pos_tag
from nltk.corpus import state_union
from nltk.corpus import stopwords

# Returns training and test files.
def init():
    train = []
    test = []
    filenames = state_union.fileids()
    for i in range(0,len(filenames)):
        if (i % 2 == 0):
            train.append(filenames[i])
        else:
            test.append(filenames[i])    
    return (train, test)

def get_stopwords():
    stop = stopwords.words()
    stop.append('applause')
    return stop

def get_speech_length(filename):
    words = state_union.words(filename)
    return len(words)

def get_vocab_size(filename):
    words = state_union.words(filename)
    vocab = set(words)
    return len(vocab)

def get_pos_tf(filename):
    words = state_union.words(filename)
    pos_words = pos_tag(words)
    freq = {}
    for (word, pos) in pos_words:
        word = word.lower()
        if word not in stop and word_regex.match(word):
            word_pos = word + '/' + pos
            if word_pos in freq:
                freq[word_pos] += 1
            else:
                freq[word_pos] = 1
    return freq

def get_tf(filename):
    words = state_union.words(filename)
    freq = {}
    for word in words:
        word = word.lower()
        if word not in stop and word_regex.match(word):
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
    return freq

def get_all_words(filenames):
    all_words = set()
    for filename in filenames:
        file_word_list = state_union.words(filename)
        file_word_set = set()
        for word in file_word_list:
            word = word.lower()
            if word not in stop and word_regex.match(word):
                file_word_set.add(word)
        all_words |= file_word_set
    return all_words

def get_idf(filenames, feature_words):
    idf = {}
    file_words = {}
    for filename in filenames:
        file_words[filename] = set(state_union.words(filename))            
    for feature_word in feature_words:
        idf[feature_word] = 0
        for filename in filenames:
            if feature_word in file_words[filename]:
                idf[feature_word] += 1
        if idf[feature_word] == 0:
            # smoothing
            idf[feature_word] = 1 
        idf[feature_word] = math.log( len(filenames) / idf[feature_word] )
    return idf

def get_tf_features(filename, feature_words):
    feature_set = {}
    file_words = get_tf(filename)
    for feature_word in feature_words:
        if feature_word in file_words:
            feature_set[feature_word] = file_words[feature_word]
        else:
            feature_set[feature_word] = 0
    return feature_set

def get_pos_tf_features(filename, feature_words):
    feature_set = {}
    file_words = get_pos_tf(filename)
    for feature_word in feature_words:
        if feature_word in file_words:
            feature_set[feature_word] = file_words[feature_word]
        else:
            feature_set[feature_word] = 0
    return feature_set

def get_tf_idf_features(filenames, feature_words):
    idf = get_idf(filenames, feature_words)
    file_features = {}
    for filename in filenames:
        feature_set = get_tf_features(filename, feature_words)
        for feature_word in feature_set:
            feature_set[feature_word] *= idf[feature_word]
        file_features[filename] = feature_set
    return file_features
        
def get_n_most_common_pos_words(filename, n):
    pos_tf = get_pos_tf(filename)
    return nlargest(n, pos_tf.items(), key=lambda item: (item[1], item[0]))
        
def get_n_most_common_words(filename, n):
    tf = get_tf(filename)
    return nlargest(n, tf.items(), key=lambda item: (item[1], item[0]))

def get_n_highest_tf_idf_words_for_each_file(filenames, n):
    print 'Getting all words...'
    all_words = get_all_words(filenames)
    print 'Calculating idf...'
    idf = get_idf(filenames, all_words)
    print 'Calculating tf-idf...'
    tf_idf_feature_words = set()
    for filename in filenames:
        tf = get_tf(filename)
        tf_idf = {}
        for word in tf:
            tf_idf[word] = tf[word] * idf[word]
        file_tf_idf_words = nlargest(n, tf_idf.items(), 
                                     key=lambda item: (item[1], item[0]))
        for tuple in file_tf_idf_words:
            tf_idf_feature_words.add(tuple[0])
    return tf_idf_feature_words

def get_50_year_class(filename):
    year = int(filename[0:4])
    if year < 1840:
        return '1790'
    elif year >= 1840 and year < 1890:
        return '1840'
    elif year >= 1890 and year < 1940:
        return '1890'
    elif year >= 1940 and year < 1990:
        return '1940'
    elif year >= 1990:
        return '1990'

def get_25_year_class(filename):
    year = int(filename[0:4])
    if year < 1815:
        return '1790'
    elif year >= 1815 and year < 1840:
        return '1815'
    elif year >= 1840 and year < 1865:
        return '1840'
    elif year >= 1865 and year < 1890:
        return '1865'
    elif year >= 1890 and year < 1915:
        return '1890'
    elif year >= 1915 and year < 1940:
        return '1915'
    elif year >= 1940 and year < 1965:
        return '1940'
    elif year >= 1965 and year < 1990:
        return '1965'
    elif year >= 1990:
        return '1990' 
    
def get_40_year_class(filename):
    year = int(filename[0:4])
    if year < 1830:
        return '[1790,1829]'
    elif year >= 1830 and year < 1870:
        return '[1830,1869]'
    elif year >= 1870 and year < 1910:
        return '[1870,1909]'
    elif year >= 1910 and year < 1950:
        return '[1910,1949]'
    elif year >= 1950 and year < 1990:
        return '[1950,1989]'  
    elif year >= 1990:
        return '[1990,2029]'
    
def get_40_year_values():
    return ['[1790,1829]', '[1830,1869]', '[1870,1909]', '[1910,1949]', 
            '[1950,1989]', '[1990,2029]']
    
def read_feature_words(input_filename):
    feature_words = set()
    feature_word_file = open(input_filename, 'r')
    for line in feature_word_file:
        word = line[:-1]
        feature_words.add(word)
    feature_word_file.close() 
    return feature_words

def read_tf_idf_feature_words():
    return read_feature_words('tf_idf_feature_words.txt')

def read_tf_feature_words():
    return read_feature_words('tf_feature_words.txt')

def read_pos_tf_feature_words():
    return read_feature_words('pos_tf_feature_words.txt')

def calculate_precision_and_recall(test_set, classifier):
    true_pos = {}
    false_neg = {}
    false_pos = {}
    for years in get_40_year_values():
        true_pos[years] = 0
        false_neg[years] = 0
        false_pos[years] = 0
    for (features, actual) in test_set:
        guess = classifier.classify(features)
        for years in get_40_year_values():
            if actual == years:
                if guess == years:
                    true_pos[years] += 1
                else:
                    false_neg[years] += 1
            else:
                if guess == years:
                    false_pos[years] += 1
    for years in get_40_year_values():
        precision = true_pos[years] / (true_pos[years] + false_pos[years])
        recall = true_pos[years] / (true_pos[years] + false_neg[years])
        f1 = 2 * precision * recall / (precision + recall)
        out = years + ' : (Precision=' + str(precision)
        out += ', Recall=' + str(recall) + ', F1=' + str(f1) + ')'
        print out
        
word_regex = re.compile('^[a-z]+$')
stop = get_stopwords()
(train, test) = init()