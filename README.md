# Active Learning for Part of Speech Tagging
This repository holds the code for my thesis on "Active Learning and Part of Speech Tagging" and specifically the part about active learning.

This is the second of two parts of code, and explores different active learning algorithms used on Part of Speech Taggers. The first part on different pos tagging models can be found on https://github.com/Xenonas/Part-of-Speech-Tagging-Multiple-Models.

After downloading the files, you need to also download word2vec pretrained model for english from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g if you wish to use information densiry algorithm.

In order to run the code you need <b>Python version 8.8.9</b> or newer and installing the requirements listed on <b>requirements.txt</b>.

The algorithms explored are:
  - Uncertainty Sampling
    - Sampling least certain instance
    - Sampling least certain sentence
    - Sampling highest certainty difference per sentence
    - Sampling sentence with highest entropy
  - Query by Committee
  - Information Density
  
  Run <b>main.py</b> in order to choose language and algorithm, in ordere to see accuracy and training history per algorithm in the Universal Dependencies datasets, EWT (english) and GDT (greek).
  
