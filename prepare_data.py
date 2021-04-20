"""
File: prepare_data.py
--------------------
Given a csv of clean/unclean text, sorts it for model training.

"""
import os
import glob
import re
import pandas as pd
import numpy as np
import textdistance
import string
from process_coha import clean_text_v2
from settings import TEXT_DIRECTORY, COMB_DIRECTORY

#OLD_CSV_FILE = "times-1820-1004.csv" # csv to be processed
CSV_DIRECTORY = "./csv-files/" # location of the csv files

ENGLISH_LEXICON = "./english-words.txt"
COMMON_ENG_LEXICON = "./google-10000-english-no-swears.txt"

"""
Plain text filenames are stored in the following format:
{sort/unsort}_{first three words of title} : {original xml file}

Combined text filenames are stored in the following format:
{first three words of title} : {original xml file}
"""

SORT_TEXT_DIRECTORY = TEXT_DIRECTORY + "raw-text/sort/" # stores sorted plain text files
UNSORT_TEXT_DIRECTORY = TEXT_DIRECTORY + "raw-text/unsort/" # stores unsorted plain text files


ERROR_MARGIN = .9 # error allowed between an unsorted and sorted word

def main():
    for csv_file in glob.glob(os.path.join(CSV_DIRECTORY, '*.csv')):
        df = pd.read_csv(csv_file)
        filenames = []
        df.apply(lambda row: sort_data(row, filenames),axis=1,result_type='expand')

    all_text = open(TEXT_DIRECTORY + "ALL_TEXT.txt", "w+")

    # prepares all_text
    all_text.write(" \t \n")

    common_eng_words = open(COMMON_ENG_LEXICON)
    for i in range(3):
        for line in common_eng_words:
            line = line.strip()
            all_text.write(str(line) + "\t" + str(line) + "\n")

    all_eng_words = open(ENGLISH_LEXICON)
    for line in all_eng_words:
        line = line.strip()
        all_text.write(str(line) + "\t" + str(line) + "\n")

    
    for f in filenames:
        unsorted_text = []
        sorted_text = []

        unsort_sent = [strip_extraneous(line) for line in open(UNSORT_TEXT_DIRECTORY + "unsort_" + f, "r")]
        sort_sent = [clean_text(line) for line in open(SORT_TEXT_DIRECTORY + "sort_" + f, "r")]

        for i in range(min(len(unsort_sent), len(sort_sent))):

            u_adjust = 0 # these adjust the indices so in case there are extra words somewhere in a list
            s_adjust = 0 

            unsort_line = unsort_sent[i].split()
            sort_line = sort_sent[i].split()

            for j in range(min(len(unsort_line), len(sort_line))):
                # checks if words are close enough matches
                #print("SIZE " + str(len(unsort_line)) + ", " + str(len(sort_line)) + " U " + str(j+u_adjust) + " J " + str(j+s_adjust))

                orig_sim = textdistance.levenshtein.normalized_similarity(unsort_line[j+u_adjust], sort_line[j+s_adjust])
                orig_dist = textdistance.levenshtein(unsort_line[j+u_adjust], sort_line[j+s_adjust])
                if (orig_sim > ERROR_MARGIN):
                    unsorted_text.append(unsort_line[j])
                    sorted_text.append(sort_line[j])
                else:
                    if ((j+u_adjust+1) < min(len(unsort_line), len(sort_line))) and ((j+s_adjust +1) <  min(len(unsort_line), len(sort_line))):
                        u_skipOne_sim = textdistance.levenshtein.normalized_similarity(unsort_line[j+u_adjust+1], sort_line[j+s_adjust])
                        s_skipOne_sim = textdistance.levenshtein.normalized_similarity(unsort_line[j+u_adjust], sort_line[j+s_adjust+1])
                        u_skipOne_dist = textdistance.levenshtein(unsort_line[j+u_adjust+1], sort_line[j+s_adjust])
                        s_skipOne_dist = textdistance.levenshtein(unsort_line[j+u_adjust], sort_line[j+s_adjust+1])
                        
                        if ((u_skipOne_sim > ERROR_MARGIN) or (s_skipOne_sim > ERROR_MARGIN)):
                            # double check this
                            if ((u_skipOne_dist) < 1) or ((s_skipOne_dist) < 1):
                                if (max(u_skipOne_sim, s_skipOne_sim) > orig_sim):
                                    if (u_skipOne_sim > s_skipOne_sim):
                                        s_adjust -= 1
                                    else:
                                        u_adjust -= 1
                                
            comb_file = open(COMB_DIRECTORY + "comb_" + f, "w+")
            for i in range(len(unsorted_text)):
                if (textdistance.levenshtein.normalized_similarity(unsorted_text[i], sorted_text[i]) > 0.7):
                    comb_file.write(unsorted_text[i] + "\t" + sorted_text[i] + "\n")
            comb_file.close()

            for i in range(len(unsorted_text)):
                if (textdistance.levenshtein.normalized_similarity(unsorted_text[i], sorted_text[i]) > 0.7):
                        all_text.write(unsorted_text[i] + "\t" + sorted_text[i] + "\n")

        print("Created File: " + f)
        print('Added to "ALL_TEXT" file: ' + f)
        lineCounter = 1
        accurateCounter = 1
        for line in open(TEXT_DIRECTORY + "ALL_TEXT.txt", "r"):
            lineCounter += 1
            if (textdistance.levenshtein.normalized_similarity(line.split("\t")[0], line.split("\t")[1]) > 0.7):
                accurateCounter += 1
        
        print("Overall Accuracy: " + str(accurateCounter/lineCounter) + "\n")
    all_text.close()



def sort_data(row: pd.Series, filenames: list) -> None:
    """ Prepares two text files from the sorted xml columns """
    if type(row["text_clean"]) == float: # checks that the field isn't empty
        return
    filename = "".join(row["title"].split()[0:3]) + " : " + row["file_name"] + ".txt"
    unsort_file = open(UNSORT_TEXT_DIRECTORY + "unsort_" + filename, "w+")
    sort_file = open(SORT_TEXT_DIRECTORY + "sort_" +filename, "w+")

    unsort_lines = strip_extraneous(row["text"]).split(".")
    sort_lines = strip_extraneous(row["text_clean"]).split(".")

    # removes sentences less than 5 words. (prevents errors later on)
    for line in unsort_lines:
        if (len(line.split()) < 5):
            unsort_lines.remove(line)
    for line in sort_lines:
        if (len(line.split()) < 5):
            sort_lines.remove(line)

    # This code assumes that there are extra periods in the unsorted text and corrects for them
    correction = 0
    for i in range(len(sort_lines)):
        if ((i + correction + 1) >= len(unsort_lines)):
                print("\nINITIAL INDEX OUT OF RANGE \n")
                break
        uString = unsort_lines[i + correction]
        sString = sort_lines[i]

        print("OUTSIDE LOOP: " + str(textdistance.levenshtein.normalized_similarity(uString, sString)))

        
        while(textdistance.levenshtein.normalized_similarity(uString, sString) < 0.7):
            print("INSIDE LOOP: " + str(textdistance.levenshtein.normalized_similarity(uString, sString)))
            if ((i + correction + 1) >= len(unsort_lines)):
                print("\nINNER INDEX OUT OF RANGE\n")
                break
            if (textdistance.levenshtein.normalized_similarity(uString + unsort_lines[i+correction], sString) < textdistance.levenshtein.normalized_similarity(uString, sString)):
                break

            correction += 1
            uString += unsort_lines[i+correction]
        
        unsort_file.write(uString + "\n")
        sort_file.write(sString + "\n")

    filenames.append(filename)
    print("Created File: " + filename)
    return


def strip_extraneous(text: str) -> str:
    """" Removes extra trailing characters from the text """
    text.strip("")
    text = re.sub('\n', "", text)
    text = re.sub('\t', "", text)
    return text


def clean_text(text: str) -> str:
    """Remove unwanted characters and extra spaces from the text"""
    text = text.lower()
    text = strip_extraneous(text)
    text = text.replace('- ','') # gets rid of line breaks

    # removes unwanted characters (i.e., unicode)
    text = text.replace('-', ' ')
    text = re.sub('[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
    return text


if __name__ == "__main__":
    main()