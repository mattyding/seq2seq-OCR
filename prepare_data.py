"""
File: prepare_data.py
--------------------
Given a text file of unsorted text, cleans it up for data preparation.
The returned text is not entirely accurate. There may be typos or char
misrecognitions
"""
import re
import pandas as pd
import numpy as np
import textdistance

CSV_FILE = "times-1820-1004.csv" # csv with rows of sorted/unsorted text

"""
Plain text names are stored in the following format:
{sort/unsort}_{first three words of title} : {original xml file}

Combined text names are stored in the following format:
{first three words of title} : {original xml file}
"""

TEXT_DIRECTORY = "./text-from-csv/" # folder where all the text files are to be stored
SORT_TEXT_DIRECTORY = TEXT_DIRECTORY + "raw-text/sort/" 
UNSORT_TEXT_DIRECTORY = TEXT_DIRECTORY + "raw-text/unsort/"
COMB_DIRECTORY = "./text-from-csv/combined-text/" #place to store combined data for training

ERROR_MARGIN = .9 # error allowed between an unsorted and sorted word (i.e., .7 similarity)

def main():
    df = pd.read_csv(CSV_FILE)
    filenames = []
    df.apply(lambda row: sort_data(row, filenames),axis=1,result_type='expand')

    all_text = open(TEXT_DIRECTORY + "ALL_TEXT.txt", "w+")
    
    for f in filenames:
        unsorted_text = []
        sorted_text = []

        u_adjust = 0 # these adjust the indices so in case there are extra words somewhere in a list
        s_adjust = 0 

        # splits line into words
        unsort_line = strip_extraneous(open(UNSORT_TEXT_DIRECTORY + "unsort_" + f).readline()).split()
        sort_line = clean_text(open(SORT_TEXT_DIRECTORY + "sort_" + f).readline()).split()

        for j in range(min(len(unsort_line), len(sort_line))):
            # checks if words are close enough matches
            print("SIZE " + str(len(unsort_line)) + ", " + str(len(sort_line)) + " U " + str(j+u_adjust) + " J " + str(j+s_adjust))

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
            comb_file.write(unsorted_text[i] + "\t" + sorted_text[i] + "\n")
        comb_file.close()
        print("Created File: " + f)

        for i in range(len(unsorted_text)):
            all_text.write(unsorted_text[i] + "\t" + sorted_text[i] + "\n")
        print('Added to "ALL_TEXT" file: ' + f)


def sort_data(row: pd.Series, filenames: list) -> None:
    """ Prepares two text files from the sorted xml columns """
    if type(row["text_clean"]) == float: # checks that the field isn't empty
        return
    filename = "".join(row["title"].split()[0:3]) + " : " + row["file_name"] + ".txt"
    unsort_file = open(UNSORT_TEXT_DIRECTORY + "unsort_" + filename, "w+")
    sort_file = open(SORT_TEXT_DIRECTORY + "sort_" +filename, "w+")

    unsort_text = row["text"]
    sort_text = row["text_clean"]

    unsort_file.write(strip_extraneous(unsort_text))
    sort_file.write(strip_extraneous(sort_text))

    filenames.append(filename)
    return


def strip_extraneous(text: str) -> str:
    """" Removes extra trailing characters from the text """
    text.strip("")
    text = re.sub('\n', "", text)
    text = re.sub('\t', "", text)
    return text


def clean_text(text: str) -> str:
    """Remove unwanted characters and extra spaces from the text"""
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


def close_match(wordUnsort: str, wordSort: str) -> bool:
    # if the length of the words is too far apart
    if abs(len(wordUnsort) - len(wordUnsort)) > (len(wordSort) * ERROR_MARGIN):
        return False
    
    # else if the letters of the words are too different
    tempWord = wordUnsort
    wrong_letters = 0
    for letter in wordSort:
        if letter not in tempWord:
            wrong_letters += 1
        else:
            tempWord = tempWord[0:tempWord.index(letter)-1] + tempWord[tempWord.index(letter)+1:]
    if (wrong_letters / len(wordSort)) > ERROR_MARGIN:
        return False
    
    return True


if __name__ == "__main__":
    main()