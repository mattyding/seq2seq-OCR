# Seq2Seq Model to Correct OCR Errors

## Abstract (pasted from writeup):
Documents transcribed with OCR software often have misreadings and errors that may make the text unsuitable for analysis. If they occur infrequently enough, it is easy to correct these errors using a combination of spell-checking tools and contextual natural language processing (NLP) models (e.g., Google BERT). However, when it comes to extremely noisy text, these strategies are often less successful.

The goal of this project was to develop a deep learning model that corrects common OCR errors. The model itself has a LSTM seq2seq architecture, commonly used in machine translation tasks. It is fed a noisy string of characters and outputs a predicted word. Out of several attempts, we found that the most effective strategy was to train the model using a historical English corpus with forced errors. These errors consist of common letter substitutions observed in OCR'ed text that are stochastically forced into the training data at a frequency proportional to their observed occurrence. Applying the model to London Times articles from 1820-1939, we were able to increase the readibility by an average of 5-10%. These results illustrate how NLP models can be trained to process noisy datasets, even when the noise hinders context-dependent tools.

## Instructions for Use:
I would suggest looking at the [example-notebooks](example-notebooks/) directory for several examples of how to use the seq2seq model. The [Basic Usage](example-notebooks/basic_usage.ipynb) provides a broad overview of correcting text with the model and is probably the best notebook to start off with.  

## File Overview:
The model is stored in the [s2s](s2s/) directory and can be accessed via the Seq2SeqOCR Class defined in [seq2seqocr.py](seq2seqocr.py).  

The training data is stored in [training-sets](training-sets/). The training script is [train_model.py](train_model.py). Other files (source data, lexicons, error probabilities) are in [source-data](training-sets/source-data/).  

The [process_letter_sub.py](process_letter_sub.py) file is for preparing OCR error probabilities from Ted Underwood's data. Those error probabilities are used in [prepare_training.py](prepare_training.py) to force errors into our training data. The [process_lexicons.py](process_lexicons.py) file saves several hashsets to disk that are used in preprocessing. Finally, [general_util.py](general_util.py) provides some general string functions shared between files.

## Citations:
seq2seq model inspired from [Keras's sample program](https://git.io/JOwXq).  
Noise function inspired by [Spell Checker](https://git.io/Jusuo).  
OCR Error data from [Ted Underwood's OCR rulesets](https://git.io/Juadv).
<br/>
Training data of historical English text taken from the following sources: [COHA](https://www.english-corpora.org/coha/), [Hansard](https://hansard.parliament.uk/), [Google's 10000 frequent English words](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html).  

Note: I manually processed Google's document to remove modern words such as "honda" and "programmer", short word-abbreviations such as "uk" and "ca", and inappropriate words.