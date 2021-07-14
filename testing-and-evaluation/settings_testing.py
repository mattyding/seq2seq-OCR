"""
settings.py
-----------
This file contains the settings for training the model.
"""

# adjusting directories so that scripts can run when CWD is seq2seqOCR. 
dirname = __file__[:-len("settings_v2.py")]

"""
Training Settings
"""
BATCH_SIZE = 64  # batch size for training.
EPOCHS = 100  # number of epochs to train for.
LATENT_DIM = 256  # latent dimensionality of the encoding space.
NUM_SAMPLES = 100000  # number of samples to train on.
DATA_PATH = dirname + "/./data/training-sets/ALL_TEXT.txt" # path to data text file; "./" indicates current folder area

BREAK_CHAR = "\t" # seperator character in data

"""
Directory Settings
"""
DOC_DIRECTORY = dirname + "/../testing_and_evaluation/text-to-predict/"  # directory containing text to predict

V1_PREDICTED_DIRECTORY = dirname + "/../testing_and_evaluation/predicted-text/model_v1/"
V2_PREDICTED_DIRECTORY = dirname + "/../testing_and_evaluation/predicted-text/model_v2/"

FREQ_DIRECTORY = dirname + "/../testing_and_evaluation/word-freq-reports/"
ACCUR_DIRECTORY = dirname + "/../testing_and_evaluation/accuracy-reports/model_v2/"

ENGLISH_LEXICON = dirname + "/./data/english-words.txt"  # lexicon of English words, complied from COHA/Google data
COMMON_ENG_LEXICON = dirname + "/./data/source-data/google-10000-english-no-swears.txt"
COHA_DIRECTORY = dirname + "/./data/source-data/COHA-sample-data/"
LETTER_SUB_DIRECTORY = dirname + "/./data/source-data/letter-conversions.txt"