"""
settings.py
-----------
This file contains the settings for evaluating versions of the trained model

"""
# adjusting directories so that scripts can run, regardless of CWD
dirname = __file__[:-len("settings.py")]


"""
Directory Navigation
"""
# INFERENCE RESULTS
DOC_DIRECTORY = dirname + "testing-and-evaluation/text-to-predict/"  # text to predict
PREDICTED_DIRECTORY = dirname + "testing-and-evaluation/predicted-text/"  # contains model results
FIGURE_DIRECTORY = dirname + "testing-and-evaluation/figures/"  # contains figures

# ENGLISH LEXICONS
COHA_DIRECTORY = dirname + "source-data/COHA-sample-data/"  # COHA samples
ENGLISH_LEXICON = dirname + "source-data/english-hashset.txt"  # words gathered from COHA, no repeats
ENGLISH_LEXICON_PICKLED = dirname + "source-data/english-hashset.pkl"
COMMON_ENG_LEXICON = dirname + "source-data/google-10000-english.txt"  # Google's 10,000 common English words
COMMON_ENG_LEXICON_PICKLED = dirname + "/source-data/google-common-english.pkl"

# TESTING RESULTS
RECOG_EVAL_DIRECTORY = dirname + "testing-and-evaluation/recognition-eval/"