"""
File: general_util.py
---------------------
This file contains util functions used in multiple files.
"""
import sys
import re

def print_statusline(msg: str):
    """
    This is a useful function for printing the current file being parsed.

    Source: https://stackoverflow.com/questions/5419389/how-to-overwrite-the-previous-print-to-stdout-in-python
    """
    last_msg_length = len(print_statusline.last_msg) if hasattr(print_statusline, 'last_msg') else 0
    print(' ' * last_msg_length, end='\r')
    print(msg, end='\r')
    sys.stdout.flush()
    print_statusline.last_msg = msg

alpha_regex_no_spaces = re.compile('[^a-zA-Z]', re.UNICODE)
def clean_text_no_spaces(text : str) -> str:
    """
    Removes all non-alphabetical characters from the input text.
    """
    return alpha_regex_no_spaces.sub('', text)

alpha_regex_with_spaces = re.compile('[^a-zA-Z ]')
def clean_text(text : str) -> str:
    """
    Removes all non-alphabetical characters from the input text.
    """
    return alpha_regex_with_spaces.sub('', text)
