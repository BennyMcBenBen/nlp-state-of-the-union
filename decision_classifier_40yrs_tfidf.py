'''
Created on Apr 25, 2010

@author: Ben
'''
from nltk import classify
from nltk.classify import DecisionTreeClassifier
from edu.zoller.nlp import common      

print 'Reading feature words...'
feature_words = common.read_tf_idf_feature_words()

print 'Assembling training feature sets...'
train_set = []
file_features = common.get_tf_idf_features(common.train, feature_words)
for filename in common.train:
    year_class = common.get_40_year_class(filename)
    train_set.append((file_features[filename], year_class))
    
print 'Training classifier...'
classifier = DecisionTreeClassifier.train(train_set)
    
print 'Assembling test feature sets...'
test_set = []
file_features = common.get_tf_idf_features(common.test, feature_words)
for filename in common.test:
    year_class = common.get_40_year_class(filename)
    test_set.append((file_features[filename], year_class))

print 'Classifying test accuracy'
print classify.accuracy(classifier, test_set)