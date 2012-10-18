'''
Created on Apr 25, 2010

@author: Ben
'''
from nltk import classify
from nltk.classify import DecisionTreeClassifier
from edu.zoller.nlp import common      

print 'Reading feature words...'
feature_words = common.read_tf_feature_words()

print 'Assembling training feature sets...'
train_set = []
for filename in common.train:
    year_class = common.get_40_year_class(filename)
    features = common.get_tf_features(filename, feature_words)
    train_set.append((features, year_class))
    
print 'Training classifier...'
classifier = DecisionTreeClassifier.train(train_set)
    
print 'Assembling test feature sets...'
test_set = []
for filename in common.test:
    year_class = common.get_40_year_class(filename)
    features = common.get_tf_features(filename, feature_words)
    test_set.append((features, year_class))

print 'Classifying test accuracy'
print classify.accuracy(classifier, test_set)