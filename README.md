# Active Learning for Part of Speech Tagging
This repository holds the code for my thesis on "Active Learning and Part of Speech Tagging" and specifically the part about active learning.

This is the second of two parts of code, and explores different active learning algorithms used on Part of Speech Taggers. The first part is on different pos tagging models and can be found on https://github.com/Xenonas/Part-of-Speech-Tagging-Multiple-Models.

After downloading the files, you need to also download word2vec pretrained model for english from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g if you wish to use information density algorithm.

In order to run the code you need <b>Python version 8.8.9</b> or newer and installing the requirements listed on <b>requirements.txt</b>.

You'll also need to install the greek package from the spacy library in order to use the greek word vectoriser. To do that, open the terminal and write:
> python -m spacy download el_core_news_sm

The algorithms explored are:
  - Uncertainty Sampling
    - Sampling least certain instance
    - Sampling least certain sentence
    - Sampling highest certainty difference per sentence
    - Sampling sentence with highest entropy
  - Query by Committee
  - Information Density

  Run <b>main.py</b> in order to choose language and algorithm, to see accuracy and training history per algorithm in the Universal Dependencies datasets, EWT (english) and GDT (greek) (https://universaldependencies.org/).
  After running main and choosing a model, if it has been used in the past, it will be loaded, if not, then the model will be trained from scratch. Then, the user can input sentences to be tagged.
  
  The code uses a simple <b>Bi-LSTM model</b>.
  
  Accuracies achived in 20 batches of 10 sentences:
  
|                                 |   Greek               |    English          
| ------------------------------- | --------------------- | ------------------      
| Random Sampling                 | 0.7770678400993347    | 0.8916313648223877  
| Least Certain Instance          | 0.8627790212631226    | 0.9475264549255371  
| Least Certain Sentence          | 0.9178774356842041    | 0.9684215784072876
| Highest Certainty Difference    | 0.9151422381401062    | 0.9687584042549133
| Highest Entropy                 | 0.9345951676368713	  | 0.9709768891334534
| Query by Committee              | 0.8265208005905151	  | 0.8941338062286377
| Information Density             | 0.8957549333572388	  | 0.9637103080749512

