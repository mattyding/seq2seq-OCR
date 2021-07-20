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
PREDICTED_DIRECTORY = dirname + "/../testing-and-evaluation/predicted-text/model_v2/"  # place to store result


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
# SOURCE DATA FOR TRAINING MODEL V2

# FORCED OCR ERRORS
LABELED_OCR_ERRORS = dirname + "/../source-data/corrected-ocr-errors.txt"
LETTER_SUBSTITUTIONS = dirname + "/../source-data/ocr-letter-conversions.txt"

NUM_LETTER_SUB_DICT = dirname + "/./training-sets/letter_sub_num.pkl"  # count of each letter substitution
PROB_LETTER_SUB_DICT = dirname + "/./training-sets/letter_sub_prob.pkl" # probability of each letter sub
ERROR_PROB_DICT = dirname + "./training-sets/ocr_error_prob.pkl"  # the probability of OCR error for each letter

LETTER_PROB_FIG = dirname + "../testing-and-evaluation/figures/ocr_error_probability.png" # stores prob graph

# ENGLISH LEXICONS
COHA_DIRECTORY = dirname + "/../source-data/COHA-sample-data/"  # COHA samples
ENGLISH_LEXICON = dirname + "/../source-data/english-hashset.txt"  # words gathered from COHA, no repeats
ENGLISH_LEXICON_PICKLED = dirname + "/../source-data/english-hashset.pkl"
COMMON_ENG_LEXICON = dirname + "/../source-data/google-10000-english.txt"  # Google's 10,000 common English words
COMMON_ENG_LEXICON_PICKLED = dirname + "../source-data/google-common-english.pkl"

# GRAPHS AND REPORTS
FIGURE_DIRECTORY = dirname + "/../testing-and-evaluation/figures/"  # directory to store figures
FREQ_DIRECTORY = dirname + "/../testing-and-evaluation/word-freq-reports/"  # word frequency plots

