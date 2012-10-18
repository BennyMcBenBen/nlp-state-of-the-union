'''
Created on Apr 25, 2010

@author: Ben
'''
from nltk import classify
from nltk.classify import NaiveBayesClassifier
from edu.zoller.nlp import common

def get_features(filename):
    feature_set = {}
    feature_set['total_words'] = common.get_speech_length(filename)
    feature_set['vocab_size'] = common.get_vocab_size(filename)
    return feature_set

print 'Assembling training feature sets...'
train_set = []
for filename in common.train:
    year_class = common.get_40_year_class(filename)
    features = get_features(filename)
    train_set.append((features, year_class))
    
print 'Training classifier...'
classifier = NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features()
    
print 'Assembling test feature sets...'
test_set = []
for filename in common.test:
    year_class = common.get_40_year_class(filename)
    features = get_features(filename)
    test_set.append((features, year_class))

print 'Classifying test accuracy'
print classify.accuracy(classifier, test_set)