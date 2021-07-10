"""
settings.py
-----------
This file contains the settings for training the model
"""
import os

# adjusting directories so that scripts can run when CWD is seq2seqOCR. 
dirname = __file__[:-len("settings_v1.py")]

"""
Training Settings
"""
BATCH_SIZE = 64  # batch size for training.
EPOCHS = 100  # number of epochs to train for.
LATENT_DIM = 256  # latent dimensionality of the encoding space.
NUM_SAMPLES = 100000  # number of samples to train on.
DATA_PATH = dirname +  "/./data/training-sets/ALL_TEXT.txt" # path to data text file; "./" indicates current folder area

BREAK_CHAR = "\t" # seperator character in data

"""
Directory Settings
"""
CSV_DIRECTORY = dirname + "/./data/source-data/csv-files/" # location of the csv files
TEXT_DIRECTORY = dirname + "/./data/training-sets/" # folder where all the text files are to be stored
COMB_DIRECTORY = dirname + "./data/training-sets/combined-text/" #place to store combined data for training

SORT_TEXT_DIRECTORY = dirname + TEXT_DIRECTORY + "raw-text/sort/" # stores sorted plain text files
UNSORT_TEXT_DIRECTORY = dirname + TEXT_DIRECTORY + "raw-text/unsort/" # stores unsorted plain text files

DOC_DIRECTORY = dirname + "/../testing_and_evaluation/text-to-predict/"  # directory containing text to predict
PREDICTED_DIRECTORY = "/../testing_and_evaluation/predicted-text/model_v1"

FREQ_DIRECTORY = dirname + "/../testing_and_evaluation/word-freq-reports/"
ACCUR_DIRECTORY = dirname + "/../testing_and_evaluation/accuracy-reports/model_v1/"

ENGLISH_LEXICON = dirname + "/./data/english-words.txt"
COMMON_ENG_LEXICON = dirname + "/./data/source-data/google-10000-english-no-swears.txt"