# Seq2Seq Model to Correct OCR Errors

## Abstract (pasted from writeup):
OCR transcriptions of historical documents often have erroneous, misrecognized characters. These errors may make the text unsuitable for analysis. If they occur infrequently enough, it is easy to correct these errors using a combination of spell-checking tools and context-based NLP models (e.g., Google BERT). However, when it comes to extremely noisy text, these strategies are often less successful.

The goal of this project was to develop a deep learning model to correct common OCR errors. The model itself has a LSTM seq2seq architecture, commonly used in NLP tasks. It is fed a noisy string of characters and outputs a predicted word. Out of several attempts, we found that the most effective strategy was to train the model using a historical English corpus with forced errors. These errors include common letter substitutions observed in OCR'ed text that were forced into the training data at a frequency proportional to their observed occurrence. Applying the model to London Times articles from 1820-1939, we were able to increase the accuracy by 5-10% on average. These results illustrate how machine learning models can be used to improve OCR transcriptions, even when noisy text makes it hard for context-dependent tools.

## Citations:
seq2seq model inspired from Keras's sample program: https://git.io/JOwXq

noise function inspired by: https://github.com/Currie32/Spell-Checker

Training data of historical English text taken from the following sources:  
COHA: https://www.english-corpora.org/coha/  
Hansard: https://hansard.parliament.uk/  
Google 10000 English Words: http://storage.googleapis.com/books/ngrams/books/datasetsv2.html  
Note: I manually processed Google's document to remove modern words such as "honda" and "programmer", uncommon short words such as "em" and "ca", and inappropriate words.