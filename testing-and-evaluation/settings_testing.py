"""
settings_vtesting.py
-----------
This file contains the settings for evaluating versions of the trained model

"""
# adjusting directories so that scripts can run, regardless of CWD
dirname = __file__[:-len("settings_testing.py")]


"""
Model Training Settings
"""
BATCH_SIZE = 64  # batch size for training.
EPOCHS = 100  # number of epochs to train for.
LATENT_DIM = 256  # latent dimensionality of the encoding space.
NUM_SAMPLES = 500000  # number of samples to train on.
BREAK_CHAR = "\t" # seperator character in data

DATA_PATH = dirname + "/./training-sets/ALL_TEXT.txt" # path to data text file; "./" indicates current folder area
SAVED_MODEL = dirname + "/./s2s-v2/"

"""
Directory Navigation
"""
# INFERENCE RESULTS
DOC_DIRECTORY = dirname + "/./text-to-predict/"  # text to predict
PREDICTED_DIRECTORY = dirname + "/./predicted-text/"  # contains model results
FIGURE_DIRECTORY = dirname + "/./figures/"  # contains figures

# ENGLISH LEXICONS
COHA_DIRECTORY = dirname + "/../source-data/COHA-sample-data/"  # COHA samples
ENGLISH_LEXICON = dirname + "/../source-data/english-hashset.txt"  # words gathered from COHA, no repeats
ENGLISH_LEXICON_PICKLED = dirname + "/../source-data/english-hashset.pkl"
COMMON_ENG_LEXICON = dirname + "/../source-data/google-10000-english.txt"  # Google's 10,000 common English words
COMMON_ENG_LEXICON_PICKLED = dirname + "../source-data/google-common-english.pkl"

# TESTING RESULTS
RECOG_EVAL_DIRECTORY = dirname + "/./recognition-eval/"