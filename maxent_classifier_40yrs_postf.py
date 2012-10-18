'''
Created on Apr 25, 2010

@author: Ben
'''
from nltk import classify
from nltk.classify import MaxentClassifier
from edu.zoller.nlp import common      

print 'Reading feature words...'
feature_words = common.read_pos_tf_feature_words()

print 'Assembling training feature sets...'
train_set = []
for filename in common.train:
    print filename
    year_class = common.get_40_year_class(filename)
    features = common.get_pos_tf_features(filename, feature_words)
    train_set.append((features, year_class))
    
print 'Training classifier...'
classifier = MaxentClassifier.train(train_set, 'CG')
classifier.show_most_informative_features(20)
    
print 'Assembling test feature sets...'
test_set = []
for filename in common.test:
    print filename
    year_class = common.get_40_year_class(filename)
    features = common.get_pos_tf_features(filename, feature_words)
    test_set.append((features, year_class))

print 'Classifying test accuracy'
print classify.accuracy(classifier, test_set)

print 'Calculating precision and recall...'
common.calculate_precision_and_recall(test_set, classifier)