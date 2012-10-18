'''
Created on Apr 25, 2010

@author: Ben
'''
from edu.zoller.nlp import common

print 'Getting feature words...'
feature_words = common.get_n_highest_tf_idf_words_for_each_file(common.train, 10)

print 'Writing feature words...'
feature_word_file = open('tf_idf_feature_words.txt', 'w')
for word in sorted(feature_words):
    feature_word_file.write(word + '\n')
feature_word_file.close()
print 'Done'
