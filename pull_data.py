import pandas as pd
import numpy as np


FILE_NAME = "times-1820-1004.csv" # csv with rows of sorted/unsorted text
DIRECTORY = "./text_from_csv/" # folder where all the text files are to be stored

def main():
    df = pd.read_csv(FILE_NAME)

    df.apply(lambda row: sort_data(row),axis=1,result_type='expand')

    print("DONE")
    return 0
    

def sort_data(row: pd.Series) -> None:
    if type(row["text_clean"]) == float: # checks that the field isn't empty
        return

    """
    File names are stored in the following format:
    {sort/unsort}_{first three words of title} : {original xml file}
    """
    filename = "".join(row["title"].split()[0:3]) + " : " + row["file_name"] + ".txt"
    unsort_file = open(DIRECTORY + "unsort_" + filename, "w+")
    sort_file = open(DIRECTORY + "sort_" +filename, "w+")

    unsort_file.write(row["text"])
    sort_file.write(row["text_clean"])
    

if __name__ == "__main__":
    main()