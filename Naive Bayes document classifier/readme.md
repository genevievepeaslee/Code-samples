Train a Naive Bayes document classification model (multivariate Bernoulli) from a corpus and classify test documents.  
\
\
To run:\
``python NB_classifier.py training_data test_data class_prior_delta cond_prob_delta model_file results``

Training and test data format:\
  ``<true class of doc1> word1:count word2:count word3:count ...``\
  ``<true class of doc2> word2:count word2:count word3:count ...``

``class_prior_delta`` is the smoothing factor for the class prior probabilities.
  P(class_i) = (num docs in class_i + class_prior_delta) / (total docs + (number of classes)*(class_prior_delta))

``cond_prob_delta`` is the smoothing factor for the conditional probabilities (word given class)
  P(word_i|class_i) = (num docs containing word_i in class_i + cond_prob_delta) / (num docs in class_i + (2 * cond_prob_delta))

Calculated probabilities of features (words) given each class are written to ``model_file``.

Results on training and test documents are written to ``results``. 

Accuracy information written directly to sys_out.
