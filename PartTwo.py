import re
import pandas as pd
import spacy
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import SnowballStemmer
#from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

#Question a
csv_path = Path.cwd() / "p2-texts" / "hansard40000.csv"

df = pd.read_csv(csv_path)

df['party'] = df['party'].replace('Labour (Co-op)', 'Labour') #1
df = df[df['party'] != 'Speaker'] #2c

party_counts = df['party'].value_counts() #2a
#print(party_counts)

sorted_counts = party_counts.sort_values(ascending=False) #2b
#print(sorted_counts)

top_parties = list(sorted_counts.index[:4]) #2c
print(f"\nTop 4 parties: {top_parties}")

df = df[df['party'].isin(top_parties)] #2d
df = df[df['speech_class'] == 'Speech'] #3
df = df[df['speech'].str.len() >= 1000] #4

"""cleaned_rows = []
for _, row in df.iterrows(): #4
    is_speech = (row['speech_class'] == 'Speech')
    long_enough = (len(row['speech']) >= 1000)
    if is_speech and long_enough:
        cleaned_rows.append(row)
cleaned_df = pd.DataFrame(cleaned_rows).reset_index(drop=True)
return cleaned_df"""

#new_party_counts = df['party'].value_counts().sort_values(ascending=False) 
#print(new_party_counts)
#print(df.shape)

"""#Question b
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['speech'])
y = df['party']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=26
)

# Print out the resulting shapes and class distributions
print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  → Training set: {X_train.shape[0]} samples")
print(f"  → Test set:     {X_test.shape[0]} samples\n")

print("Class distribution in training set:")
print(y_train.value_counts())
print("\nClass distribution in test set:")
print(y_test.value_counts())

#question c

rf_clf  = RandomForestClassifier(n_estimators=300, random_state=26)
svm_clf = SVC(kernel='linear', random_state=26)

rf_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

y_pred_rf  = rf_clf.predict(X_test)
y_pred_svm = svm_clf.predict(X_test)

print("=== Random Forest (n_estimators=300) ===")
print("Macro-average F1:   ", f1_score(y_test, y_pred_rf,  average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

print("\n\n=== SVM (linear kernel) ===")
print("Macro-average F1:   ", f1_score(y_test, y_pred_svm, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

#question d

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 3))
X = vectorizer.fit_transform(df['speech'])
y = df['party']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y, #d
    random_state=26
)

# Print out the resulting shapes and class distributions
print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  → Training set: {X_train.shape[0]} samples")
print(f"  → Test set:     {X_test.shape[0]} samples\n")

print("Class distribution in training set:")
print(y_train.value_counts())
print("\nClass distribution in test set:")
print(y_test.value_counts())

rf_clf  = RandomForestClassifier(n_estimators=300, random_state=26)
svm_clf = SVC(kernel='linear', random_state=26)

rf_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

y_pred_rf  = rf_clf.predict(X_test)
y_pred_svm = svm_clf.predict(X_test)

print("=== Random Forest (n_estimators=300) ===")
print("Macro-average F1:   ", f1_score(y_test, y_pred_rf,  average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

print("\n\n=== SVM (linear kernel) ===")
print("Macro-average F1:   ", f1_score(y_test, y_pred_svm, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))"""


# Assumes at the top of your script you have imported:
#   re, SnowballStemmer, ENGLISH_STOP_WORDS,
#   TfidfVectorizer, train_test_split,
#   RandomForestClassifier, SVC, MultinomialNB, f1_score

# Initialize stemmer and stop-word set
stemmer = SnowballStemmer('english')
stop_words = ENGLISH_STOP_WORDS

# Custom tokenizer: strip punctuation & lowercase, drop stop-words, stem
def custom_tokenizer_with_stemming(text):
    # lowercase & remove punctuation
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    tokens = cleaned.split()
    # drop stop-words
    filtered = [tok for tok in tokens if tok not in stop_words]
    # stem tokens
    return [stemmer.stem(tok) for tok in filtered]

# TF-IDF vectoriser: include 1–3 grams, top 3000 features (no min_df/max_df filtering)
vectorizer = TfidfVectorizer(
    tokenizer=custom_tokenizer_with_stemming,
    token_pattern=None,
    #ngram_range=(1, 3),
    max_features=3000
)

# Fit-transform the speech texts
X = vectorizer.fit_transform(df['speech'])
# Labels
y = df['party']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=26
)

# Define classifiers with Random Forest using 50 trees
classifiers = {
    "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=26),
    "Linear SVM"              : SVC(kernel='linear', random_state=26),
    "Multinomial NB"          : MultinomialNB()
}

# Train & print macro-F1 for each
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name:<25} macro-F1 = {f1_score(y_test, y_pred, average='macro'):.3f}")

print("bigrams and trigrams, stemming, stop-words removed, punct + lower removal included")