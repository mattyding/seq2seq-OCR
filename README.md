# Seq2Seq Model to Correct OCR Errors

## Abstract (pasted from writeup):
OCR transcriptions of historical documents often have erroneous, misrecognized characters. These errors may make the text unsuitable for analysis. If they occur infrequently enough, it is easy to correct these errors using a combination of spell-checking tools and context-based NLP models (e.g., Google BERT). However, when it comes to extremely noisy text, these strategies are often less successful.

The goal of this project was to develop a deep learning model to correct common OCR errors. The model itself has a LSTM seq2seq architecture, commonly used in NLP tasks. It is fed a noisy string of characters and outputs a predicted word. Out of several attempts, we found that the most effective strategy was to train the model using a historical English corpus with forced errors. These errors include common letter substitutions observed in OCR'ed text that were forced into the training data at a frequency proportional to their observed occurrence. Applying the model to London Times articles from 1820-1939, we were able to increase the accuracy by 5-10% on average. These results illustrate how machine learning models can be used to improve OCR transcriptions, even when noisy text makes it hard for context-dependent tools.

## Instructions for Use:
I would recommend looking at the [example-notebooks](example-notebooks/) directory for several examples of how to use the seq2seq model. The [Basic Usage](example-notebooks/basic_usage.ipynb) provides a broad overview of correcting text with the model, and is probably the best notebook to start off with.  

## File Overview:
The model is stored in the [s2s](s2s/) directory and can be accessed via the Seq2SeqOCR Class defined in [seq2seqocr.py](seq2seqocr.py). The training data is stored in [training-sets](training-sets/). Other files (source data, lexicons, error probabilities) are in [source-data](training-sets/source-data/).

## Citations:
seq2seq model inspired from [Keras's sample program](https://git.io/JOwXq).  
Noise function inspired by [Spell Checker](https://git.io/Jusuo).  
OCR Error data from [Ted Underwood's OCR rulesets](https://github.com/tedunderwood/DataMunging/tree/master/rulesets/).
<br/>
Training data of historical English text taken from the following sources: [COHA](https://www.english-corpora.org/coha/), [Hansard](https://hansard.parliament.uk/), [Google's 10000 frequent English words](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html).  

Note: I manually processed Google's document to remove modern words such as "honda" and "programmer", uncommon short words such as "em" and "ca", and inappropriate words.