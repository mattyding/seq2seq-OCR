"""
settings_v2.py
-----------
This file contains the settings for training the second version of the correction model.

NOTE: to run the model in inference mode on a different set of text files, edit the 
DOC_DIRECTORY variable below to be a directory containing the text to be evaluated.
"""
# adjusting directories so that scripts can run, regardless of CWD
dirname = __file__[:-len("settings.py")]

"""
Model Training Settings
"""
BATCH_SIZE = 64  # batch size for training
EPOCHS = 100  # number of epochs to train
LATENT_DIM = 256  # latent dimensionality of the encoding space
NUM_SAMPLES = 500000  # number of samples to train on
BREAK_CHAR = '\t' # seperator character in data
ENDSEQ_CHAR = '\n' # denotes end of sequence for decoder

# shorter predictions are too inaccurate, so the model is only trained on >3-char words
MIN_SEQ_LENGTH = 4
MAX_SEQ_LENGTH = 14 # maximum length of input sequence

DATA_PATH = dirname + "training-sets/forced_errors.txt" # path to data directory
SAVED_MODEL = dirname + "s2s/"

"""
Directory Navigation
"""
# FORCED OCR ERRORS
LABELED_OCR_ERRORS = dirname + "training-sets/source-data/corrected_ocr_errors.txt"
LETTER_SUBSTITUTIONS = dirname + "training-sets/source-data/ocr_letter_conversions.txt"

# count of letter substitutions
NUM_LETTER_SUB_DICT = dirname + "training-sets/source-data/letter_sub_num.pkl"
# substitution probabilities per letter
PROB_LETTER_SUB_DICT = dirname + "training-sets/source-data/letter_sub_prob.pkl"
# probability of OCR error per letter
ERROR_PROB_DICT = dirname + "training-sets/source-data/ocr_error_prob.pkl"

LETTER_PROB_FIG = dirname + "figures/ocr_error_probability.png" # stores prob graph

# ENGLISH LEXICONS
COHA_DIRECTORY = dirname + "training-sets/source-data/COHA-sample-data/"  # COHA samples
ENGLISH_LEXICON = dirname + "training-sets/source-data/english_hashset.txt"  # COHA words, no repeats
ENGLISH_LEXICON_PKL = dirname + "training-sets/source-data/english_hashset.pkl"
COMMON_ENG_LEXICON = dirname + "training-sets/source-data/google_english_cleaned.txt"  # Google's 10,000 most-common English words
COMMON_ENG_LEXICON_PKL = dirname + "training-sets/source-data/common_hashset.pkl"

# GRAPHS AND REPORTS
FIGURE_DIRECTORY = dirname + "figures/"  # directory to store figures