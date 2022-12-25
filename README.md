# Active-Learning-for-Part-of-Speech-Tagging
This repository holds the code for my thesis on "Active Learning and Part of Speech Tagging" and specifically the part about active learning.

This is the second of two parts of code, and explores different active learning algorithms used on Part of Speech Taggers. 

The algorithms explored are:
  - Uncertainty Sampling
    - Sampling least certain instance
    - Sampling least certain sentence
    - Sampling highest certainty difference per sentence
    - Sampling sentence with highest entropy
  - Query by Committee
  - Information Density
  
  Run main.py in order to choose language and algorithm, in ordere to see accuracy and training history per algorithm in the Universal Dependencies datasets, EWT (english) and GDT (greek).
  
