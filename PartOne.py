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
import os 
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title, author, and year"""
    rows = []

    for file in path.glob("*.txt"):
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
    """
    
    cleaned_text = clean_text(text)  # clean the text
    tokens = nltk.word_tokenize(cleaned_text)
    sentences = nltk.sent_tokenize(text)  # split text into sentences

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
    """
    word = word.lower()

    if word in d:
        return len(d[word][0])  # return the number of syllables in the dictionary, should be the first entry of the list
    else: # estimate syllables by counting vowel clusters
        vowels = ['a','e','i','o','u','y']
        count = 0
        prev_vowel = False
        for char in word:
            if char in vowels:
                if not prev_vowel:
                    count += 1
                prev_vowel = True
            else:
                prev_vowel = False
        
    if count > 0:
        return count
    else:
        return 1 # at least one syllable per word

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    
    parsed_docs = []
    for story in df["text"]:
        if len(story) > 1000000:
            print("File too big, parsing texts in sections as per note on question 1d")
            doc_part = []
            for i in range(0, len(story), 1000000):
                section = story[i:i+1000000]
                doc_part.append(nlp(section))  # parse the section
            
            complete_doc = Doc.from_docs(doc_part)  # combine the sections into a single Doc
        else:
            complete_doc = nlp(story)  # parse the whole text

        parsed_docs.append(complete_doc)  # add the parsed doc to the list
    
    df['parsed'] = parsed_docs  # add the parsed docs to the DataFrame

    df.to_pickle(store_path / out_name)  # save the DataFrame to a pickle file

    return df


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



"""def subjects_by_verb_count(doc, verb):
    Extracts the most common subjects of a given verb in a parsed document. Returns a list.
    subjects = Counter()

    for token in doc:
        if token.lemma_ == verb and token.pos_ == "VERB":
            for subject in token.children:
                if subject.dep_ in ("nsubj", "nsubjpass"):
                    subjects[subject.lemma_] += 1

    
    return subjects.most_common(100)  # return the 100 most common subjects"""


"""def adjective_counts(doc):
    Extracts the most common adjectives in a parsed document. Returns a list of tuples.
    adjectives = Counter()

    for token in doc:
        if token.pos_ == "ADJ":
            adjectives[token.lemma_] += 1
    
    return adjectives.most_common(100)  # return the 100 most common adjectives"""

def novel_titles(df):
    """Returns the title of each novel and a list of the ten most common syntatic objects overall in the text"""
    results = {}

    for title, doc in zip(df['title'], df['parsed']):
        syntactic_objects = Counter()

        for token in doc:
            if token.dep_ in ("dobj", "pobj"):
                syntactic_objects[token.lemma_] += 1

        results[title] = syntactic_objects.most_common(10)  # get the 10 most common syntactic objects

    return results
    
def novel_hear_syntactics(df):
    """Returns the title of each novel and a list of the ten most common syntatic objects of the verb to hear (in any tense) in the text, ordered by their frequency"""
    results = {}

    for title, doc in zip(df['title'], df['parsed']):
        hear_objects = Counter()

        for token in doc:
            if token.dep_ == "VERB" and token.lemma_ == "hear":
                for object in token.children:
                    if object.dep_ in ("dobj", "pobj"):
                        hear_objects[token.lemma_] += 1

        results[title] = hear_objects.most_common(10)  # get the 10 most common syntactic objects

    return results

def novel_hear_pmis(df):
    """Returns the title of each novel and a list of the ten most common syntatic objects of the verb to hear (in any tense) in the text, ordered by their pointwise mutual information (PMI) with the verb ‘to hear’"""
    obj_cnt = Counter()
    hear_cnt = 0
    cooccur = Counter()

    for token in doc:
        if token.dep_ in ("dobj", "pobj"):
            obj_cnt[token.lemma_] += 1

            head_word = token.head
            if head_word.pos == "VERB" and head_word.lemma_ == "hear":
                hear_cnt += 1
                cooccur[token.lemma_] += 1

     # 2) Compute PMI for each object that actually occurred with hear
        N = sum(obj_cnt.values())  # total object relations in the novel
        pmi_scores = {}
        for lemma, c_xy in cooccur.items():
            c_x = obj_cnt[lemma]
            c_y = hear_cnt
            # PMI = log2((c_xy * N) / (c_x * c_y))
            pmi_scores[lemma] = math.log2((c_xy * N) / (c_x * c_y))

        # 3) Take the top 10 by descending PMI
        top10 = sorted(pmi_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
        results[title] = top10

    return results

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

