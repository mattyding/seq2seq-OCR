"""
accuracy-counter.py
-------------------
This program evaluates the accuracy of the trained model by comparing pre/post-training text 
to see how much of the pre and post trained text comprises recognizable English words.
"""
import os
import pickle
from settings import DOC_DIRECTORY, PREDICTED_DIRECTORY, RECOG_EVAL_DIRECTORY
from settings import ENGLISH_LEXICON_PICKLED


def main():
    """
    Compares each model's output to the original text and percentage of each file that can
    be recognized as English words (i.e., the percentage of words that are in ENGLISH_LEXICON_PICKLED).
    
    Saves a dict to file with the following items:
        d = {
            "TOTAL" : total_files,  # int: total number of files
            "ORIG" : orig_percent,  # list: percentage of original text that is recognized as English words
            "PRED" : pred_percent,  # list: percentage of predicted text that is recognized as English words
            "ZIP" : zip_percent,    # list: zipped file of original/predicted percentages for each file
            "FLAT" : flat_change,   # list: flat percentage change between original and predicted text
            "PERC" : perc_change,   # list: percentage change between original and predicted text
            "SIZE" : file_size      # list: number of words in each file
        }
    """
    with open(ENGLISH_LEXICON_PICKLED, "rb") as f:
        english_words = pickle.load(f)

    # adds files to evaluate to a queue (list)
    for model_folder in os.listdir(PREDICTED_DIRECTORY):

        pathsToEval = []
        for proj_folder in os.listdir(PREDICTED_DIRECTORY + model_folder):
            for root, dirs, files in os.walk(PREDICTED_DIRECTORY + model_folder + "/" + proj_folder):
                for f in files:
                    if f.endswith(".txt"):
                        pathsToEval.append(root + "/" + f)

        total_files = len(pathsToEval)
        orig_percent, pred_percent, file_size = [], [], []

        while (len(pathsToEval) > 0):
            # gets the next file to evaluate
            currPath = pathsToEval.pop()
            pathSegment = currPath[currPath.find(PREDICTED_DIRECTORY) + len(PREDICTED_DIRECTORY):]
            pathSegment = pathSegment[pathSegment.find("/") + 1:]
            lastDirec = pathSegment.rfind("/") # index of the last "/" in the path
            origPath = DOC_DIRECTORY + pathSegment[:lastDirec] + "/" + pathSegment[lastDirec + 1 + len("P_"):]
            
            orig_f = [line.split(" ") for line in open(origPath, "r").readlines()]
            pred_f = [line.split(" ") for line in open(currPath, "r").readlines()]

            accurate_orig, accurate_pred = 0, 0

            for i in range(min(len(orig_f[0]), len(pred_f[0]))):
                if orig_f[0][i] in english_words:
                    accurate_orig += 1
                if pred_f[0][i] in english_words:
                    accurate_pred += 1
            
            orig_total = len(orig_f[0])
            pred_total = len(pred_f[0])

            d = {}
            #d[filename] = [total words, overall accuracy, eng words in original, eng words in predicted, difference]
            d[f] = [pred_total, accurate_orig, accurate_pred, accurate_pred - accurate_orig]
            #print(f"ACCURACY: {textdistance.levenshtein.normalized_similarity(orig_f[0], pred_f[0])}")
            orig_percent.append(accurate_orig / orig_total)
            pred_percent.append(accurate_pred / pred_total)
            file_size.append(pred_total)

        zip_percent = list(zip(orig_percent, pred_percent))
        flat_change = [p - o for o, p in zip_percent]
        perc_change = [f / o if (o != 0) else 0 for o, f in list(zip(orig_percent, flat_change))]

        if total_files > 0:  # in case we didn't process any files
            d = {
                "TOTAL" : total_files,
                "ORIG" : orig_percent,
                "PRED" : pred_percent,
                "ZIP" : zip_percent,
                "FLAT" : flat_change,
                "PERC" : perc_change,
                "SIZE" : file_size
            }

            with open(f"{RECOG_EVAL_DIRECTORY}{model_folder}-{proj_folder}.pkl", "wb") as f:
                pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            print(f"{model_folder}-{proj_folder} processed.")

    
if __name__ == "__main__":
    main()