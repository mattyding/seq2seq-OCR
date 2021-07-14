"""
settings_v2.py
-----------
This file contains the settings for training the second version of the correction model.

NOTE: to run the model in inference mode on a different set of text files, edit the 
DOC_DIRECTORY variable below to be a directory containing the text to be evaluated.
"""
# adjusting directories so that scripts can run, regardless of CWD
dirname = __file__[:-len("settings_v2.py")]

# INFERENCE
DOC_DIRECTORY = dirname + "/../testing-and-evaluation/text-to-predict/"  # text to predict
PREDICTED_DIRECTORY = dirname + "/../testing-and-evaluation/predicted-text/model_v2"  # place to store result


"""
Model Training Settings
"""
BATCH_SIZE = 64  # batch size for training.
EPOCHS = 100  # number of epochs to train for.
LATENT_DIM = 256  # latent dimensionality of the encoding space.
NUM_SAMPLES = 100000  # number of samples to train on.
DATA_PATH = dirname + "/./training-sets/ALL_TEXT.txt" # path to data text file; "./" indicates current folder area
BREAK_CHAR = "\t" # seperator character in data

"""
Directory Navigation
"""
# SOURCE DATA FOR TRAINING MODEL V2
LETTER_SUB_DIRECTORY = dirname + "/./training-sets/ocr-letter-conversions.txt"

# ENGLISH LEXICON
ENGLISH_LEXICON = dirname + "/./training-sets/english-words.txt"
COHA_DIRECTORY = dirname + "/../source-data/COHA-sample-data/"  # COHA samples
COMMON_ENG_LEXICON = dirname + "/../source-data/google-10000-english.txt"  # Google's 10,000 common English words

# GRAPHS AND REPORTS
FREQ_DIRECTORY = dirname + "/../testing-and-evaluation/word-freq-reports/"  # word frequency plots
ACCUR_DIRECTORY = dirname + "/../testing-and-evaluation/accuracy-reports/model_v2/"  # training accuracy reports

