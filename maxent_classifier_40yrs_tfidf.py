'''
Created on Apr 25, 2010

@author: Ben
'''
from nltk import classify
from nltk.classify import MaxentClassifier
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
classifier = MaxentClassifier.train(train_set, 'CG')
classifier.show_most_informative_features()
    
print 'Assembling test feature sets...'
test_set = []
file_features = common.get_tf_idf_features(common.test, feature_words)
for filename in common.test:
    year_class = common.get_40_year_class(filename)
    test_set.append((file_features[filename], year_class))

print 'Classifying test accuracy'
print classify.accuracy(classifier, test_set)

print 'Calculating precision and recall...'
common.calculate_precision_and_recall(test_set, classifier)