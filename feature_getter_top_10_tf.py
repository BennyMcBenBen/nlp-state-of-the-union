'''
Created on Apr 25, 2010

@author: Ben
'''
from edu.zoller.nlp import common

print 'Getting feature words...'
feature_words = set()
for filename in common.train:
    for item in common.get_n_most_common_words(filename, 10):
        word = item[0]
        feature_words.add(word)
        
print 'Writing feature words...'
feature_word_file = open('tf_feature_words.txt', 'w')
for word in sorted(feature_words):
    feature_word_file.write(word + '\n')
feature_word_file.close()
print 'Done'