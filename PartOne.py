#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
import string

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

#Question 1a
#def read_novels(path=Path.cwd() / "texts" / "novels"):
#    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
#    author, and year"""
#    pass

def read_novels(novel_path):
    rows = []
    novel_path = Path(novel_path)

    for file in novel_path.glob("*.txt"):
        parts = file.stem.split("-") 

        year = int(parts[-1]) #last bit of the filename is the year
        author = str(parts[-2]) #second last bit of the filename is the author
        title_raw = str(parts[:-2]) #everything else is the title

        title = title_raw.replace("_", " ").title()  # replace underscores with spaces and title case

        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

# create an empty dataframe with the column headings we need
#evaluation_results = pd.DataFrame(columns=['Model', 'Test Words', 'Cosine Similarity'], dtype=object)

        rows.append({"text": text, "title": title, "author": author, "year": year})
        
    df = pd.DataFrame(rows)

    df = df.sort_values(by="year", ascending=True) #sort by year
    #df = df.reset_index(drop=True)  # reset index after sorting
    return df

#print(read_novels(r"C:\Users\eddie\NLP\Coursework\p1-texts\novels"))

#ttr_text_raw = read_novels(r"C:\Users\eddie\NLP\Coursework\p1-texts\novels")
#ttr_text = ttr_text_raw["text"].tolist()

def clean_text(text):
    """Cleans the text by removing punctuation and converting to lowercase."""
    text = text.lower()  # convert to lowercase
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))  # regex for punctuation
    text = re_punc.sub('', text)  # remove punctuation
    return text

def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    cleaned_text = clean_text(text)  # clean the text

    tokens = nltk.word_tokenize(cleaned_text)

    words = [word for word in tokens if word.isalpha()]  #only alphabetic tokens
    
    if not words: #handle no words
        return 0
    
    ttr = len(set(words)) / len(words)  # type-token ratio

    return ttr
    #pass


def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    
    cleaned_text = clean_text(text)  # clean the text
    tokens = nltk.word_tokenize(cleaned_text)

    words = [word for word in tokens if word.isalpha()]  # only alphabetic tokens

    total_sentences = len(sentences)
    total_words = len(words)

    if total_words == 0 or total_sentences == 0:
        return 0

    total_syllables = 0
    for word in words:
        syllables = count_syl(word, d)
        total_syllables += syllables
    
    avg_sentence_length = total_words / total_sentences
    avg_syllables_per_word = total_syllables / total_words

    fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59 #check this

    return fk_grade

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

