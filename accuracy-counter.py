import os
import textdistance
import string
import numpy as np
import matplotlib.pyplot as plt

ENGLISH_LEXICON = "./english-words.txt"
PREDICTED_TEXT_DIR = "./text-to-predict/predicted/"
RAW_TEXT_DIR = "./text-to-predict/raw-text/"

#FOLDER_NAME = "0FFO-1820-OCT04_Issue.xml" # CALCULATES THE ACCURACY OF ALL FILES IN THIS FOLDER

def main():
    english_words = set()
    for line in open(ENGLISH_LEXICON):
        english_words.add(line.strip())
    
    orig_percent = np.array([])
    pred_percent = np.array([])
    file_size = np.array([])

    total_files = 0

    for folder_name in os.listdir(RAW_TEXT_DIR):
        if folder_name == ".DS_Store": # this is a file that mac machines automatically make that gets in the way
            continue
        for f in os.listdir(RAW_TEXT_DIR + folder_name):
            if f == ".DS_Store":
                continue
            total_files += 1

            accurate_orig, accurate_pred = 0, 0

            orig_f = [line.split(" ") for line in open(RAW_TEXT_DIR + folder_name + "/" + f)]
            pred_f = [line.split(" ") for line in (open(PREDICTED_TEXT_DIR + folder_name + "/P_" +f))]
            for i in range(len(pred_f[0])):
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
            orig_percent = np.append(orig_percent, accurate_orig / orig_total)
            pred_percent = np.append(pred_percent, accurate_pred / pred_total)
            file_size = np.append(file_size, pred_total)
    
    print(f"TOTAL FILES: {total_files}\n")
    area = (0.01 * file_size)**2 * 0.5

    plt.scatter(orig_percent, pred_percent, s=area, alpha=0.5)
    plt.plot([-1, 2], [-1, 2], color="#000000", linestyle="dashed", linewidth="0.5")
    plt.xlabel("Original Accuracy")
    plt.ylabel("Predicted\nAccuracy", rotation="horizontal", loc="center")
    plt.title(f"Machine Learning OCR Prediction Accuracies (n={total_files})")
    plt.xlim(0, 1), plt.ylim(0, 1)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    gll = plt.scatter([],[], s=(0.01 * 1000)**2 * 0.5, marker='o', color='#555555')
    gl = plt.scatter([],[], s=(0.01 * 2000)**2 * 0.5, marker='o', color='#555555')
    ga = plt.scatter([],[], s=(0.01 * 4000)**2 * 0.5, marker='o', color='#555555')
    lgd = plt.legend((gll,gl,ga),
       ('   1000', '    2000', '   4000'),
       title="Words in File",
       scatterpoints=1,
       labelspacing=3,
       bbox_to_anchor=(1.05, 1), 
       loc='upper left', 
       borderaxespad=0., 
       borderpad=2,
       #loc='lower right',
       ncol=1,
       fontsize=8)._legend_box.sep = 10
    plt.savefig("accuracy-testing-size.png", bbox_inches='tight')
    """
    plt.scatter(orig_percent, pred_percent)
    plt.plot([-1, 2], [-1, 2], color="#000000", linestyle="dashed", linewidth="0.5")
    plt.xlabel("Original Accuracy")
    plt.ylabel("Predicted\nAccuracy", rotation="horizontal", loc="center")
    plt.title(f"Machine Learning OCR Prediction Accuracies (n={total_files})")
    plt.xlim(0, 1), plt.ylim(0, 1)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.savefig("accuracy-testing-normal.png", bbox_inches='tight')
    """

            
        
    
if __name__ == "__main__":
    main()