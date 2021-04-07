"""
settings.py
-----------
This file contains the settings for training the model.
"""

"""
Training Settings
"""
BATCH_SIZE = 64  # batch size for training.
EPOCHS = 100  # number of epochs to train for.
LATENT_DIM = 256  # latent dimensionality of the encoding space.
NUM_SAMPLES = 50000  # number of samples to train on.
DATA_PATH = "./text-from-csv/ALL_TEXT.txt" # path to data text file; "./" indicates current folder area

BREAK_CHAR = "\t" # seperator character in data

"""
Directory Settings
"""
TEXT_DIRECTORY = "./text-from-csv/" # folder where all the text files are to be stored
COMB_DIRECTORY = "./text-from-csv/combined-text/" #place to store combined data for training