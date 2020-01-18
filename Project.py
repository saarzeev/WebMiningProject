import csv
import re

import gensim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from keras import Input
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import Model
from keras_preprocessing.sequence import pad_sequences
from nltk import WordPunctTokenizer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from keras.preprocessing.text import Tokenizer
from tensorflow.python.estimator import keras
import tensorflow_estimator


class ExclaimTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        exclaims = pd.DataFrame(X.apply(lambda x: x.count("!")))
        return exclaims

    def fit(self, X, y=None, **fit_params):
        return self


class TimeOfDayTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        exclaims = pd.DataFrame(X.apply(lambda x: float(x.split(" ")[3].split(":")[0])))
        return exclaims

    def fit(self, X, y=None, **fit_params):
        return self


class ColumnExtractor(TransformerMixin):

    def __init__(self, col):
        self.col = col

    def transform(self, X):
        return X[self.col]

    def fit(self, X, y=None):
        return self


#
# class TimeColumnExtractor(object):
#
#     def transform(self, X):
#         col = X.date # column 3 and 4 are "extracted"
#         return col
#
#     def fit(self, X, y=None):
#         return self

def load_dict_smileys():
    return {
        ":‑)": "smiley",
        ":-]": "smiley",
        ":-3": "smiley",
        ":->": "smiley",
        "8-)": "smiley",
        ":-}": "smiley",
        ":)": "smiley",
        ":]": "smiley",
        ":3": "smiley",
        ":>": "smiley",
        "8)": "smiley",
        ":}": "smiley",
        ":o)": "smiley",
        ":c)": "smiley",
        ":^)": "smiley",
        "=]": "smiley",
        "=)": "smiley",
        ":-))": "smiley",
        ":‑D": "smiley",
        "8‑D": "smiley",
        "x‑D": "smiley",
        "X‑D": "smiley",
        ":D": "smiley",
        "8D": "smiley",
        "xD": "smiley",
        "XD": "smiley",
        ":‑(": "sad",
        ":‑c": "sad",
        ":‑<": "sad",
        ":‑[": "sad",
        ":(": "sad",
        ":c": "sad",
        ":<": "sad",
        ":[": "sad",
        ":-||": "sad",
        ">:[": "sad",
        ":{": "sad",
        ":@": "sad",
        ">:(": "sad",
        ":'‑(": "sad",
        ":'(": "sad",
        ":‑P": "playful",
        "X‑P": "playful",
        "x‑p": "playful",
        ":‑p": "playful",
        ":‑Þ": "playful",
        ":‑þ": "playful",
        ":‑b": "playful",
        ":P": "playful",
        "XP": "playful",
        "xp": "playful",
        ":p": "playful",
        ":Þ": "playful",
        ":þ": "playful",
        ":b": "playful",
        "<3": "love",
        "\o/": "cheer"
    }


# self defined contractions
def load_dict_contractions():
    return {
        "ain't": "is not",
        "amn't": "am not",
        "aren't": "are not",
        "b/c": "because",
        "can't": "cannot",
        "'cause": "because",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "could've": "could have",
        "cya": "see you",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "e'er": "ever",
        "em": "them",
        "everyone's": "everyone is",
        "finna": "fixing to",
        "fml": "fuck my life",
        "fb": "facebook",
        "gimme": "give me",
        "gonna": "going to",
        "gon't": "go not",
        "gotta": "got to",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "he've": "he have",
        "how'd": "how would",
        "how'll": "how will",
        "how're": "how are",
        "how's": "how is",
        "I'd": "I would",
        "I'll": "I will",
        "I'm": "I am",
        "I'm'a": "I am about to",
        "I'm'o": "I am going to",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "I've": "I have",
        "kinda": "kind of",
        "let's": "let us",
        "lol": "laughing out loud",
        "mayn't": "may not",
        "may've": "may have",
        "mightn't": "might not",
        "might've": "might have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "must've": "must have",
        "needn't": "need not",
        "ne'er": "never",
        "nite": "night",
        "o'": "of",
        "o'er": "over",
        "ol'": "old",
        "oughtn't": "ought not",
        "rofl": "rolling on the floor laughing",
        "shalln't": "shall not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "should've": "should have",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "something's": "something is",
        "that'd": "that would",
        "that'll": "that will",
        "that're": "that are",
        "that's": "that is",
        "there'd": "there would",
        "there'll": "there will",
        "there're": "there are",
        "there's": "there is",
        "these're": "these are",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "this's": "this is",
        "those're": "those are",
        "'tis": "it is",
        "'twas": "it was",
        "u": "you",
        "w/": "with",
        "w/o": "without",
        "wanna": "want to",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'd": "what did",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "where'd": "where did",
        "where're": "where are",
        "where's": "where is",
        "where've": "where have",
        "which's": "which is",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why's": "why is",
        "won't": "will not",
        "wouldn't": "would not",
        "would've": "would have",
        "y'all": "you all",
        "ya": "you",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "Whatcha": "What are you",
        "luv": "love",
        "sux": "sucks"
    }


SMILEY = load_dict_smileys()
CONTRACTIONS = load_dict_contractions()


def tweet_cleaner(text, tags_removal=True, links_removal=True, transform_smilies=True, transform_contractions=True):
    remove_tags = r'@[A-Za-z0-9]+'
    remove_links = r'https?://[A-Za-z0-9./]+'
    remove_tags_and_links_re = r'|'.join((remove_tags, remove_links))
    # Discard any html stuff in tweets
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    # Remove tags and links
    if tags_removal and links_removal:
        stripped = re.sub(remove_tags_and_links_re, '', souped)
    elif tags_removal:
        stripped = re.sub(remove_tags, '', souped)
    elif links_removal:
        stripped = re.sub(remove_links, '', souped)
    else:
        stripped = souped

    try:
        # Remove weird characters
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped

    if transform_smilies:
        words = clean.split()
        reformed = [SMILEY[word] if word in SMILEY else word for word in words]
        words = " ".join(reformed)

    if transform_contractions:
        tweet = words.replace("’", "'")
        words = tweet.split()
        reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
        words = " ".join(reformed)

    lower_case = words.lower()
    # preprocessing has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tokenizer.tokenize(lower_case)
    return (" ".join(words)).strip()


colnames = ['target', 'tweet_id', 'date', 'query', 'user', 'text']
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', names=colnames, header=None, engine='python',
                 error_bad_lines=False)

# Uncomment to run trainings faster on a smaller dataset
# df = df.sample(n=10000, replace=False)
tokenizer = WordPunctTokenizer()


def normalize_binary(x):
    if x == 4:
        return 1
    return 0


df['label'] = df.target.map(normalize_binary)

# df['length'] = df['text'].str.len()
# exclaimer = lambda x: x.count("!")
# pointer = lambda x: x.count(".")
# count_upper = lambda message: sum(1 for c in message if c.isupper())
# get_lower_upper_ration = lambda message: float((sum(1 for c in message if c.islower())) + 1) / (
#             sum(1 for c in message if c.isupper()) + 1)
# get_time = lambda x:  x.split(" ")[3].split(":")[0]
#
# def get_multiple(message):
#     count = 0
#     for i in range(len(message) - 1):
#         if message[i] == message[i + 1]:
#             count += 1
#
#     return (float(count) + 1) / (float(len(message)) + 1)
#
#
# df['exclaim'] = df.text.map(exclaimer)
# df['point'] = df.text.map(pointer)
# df['capital'] = df.text.map(count_upper)
# df['capital_lower_ration'] = df.text.map(get_lower_upper_ration)
# df['multiple'] = df.text.map(get_multiple)
# df['time'] = df.date.map(get_time)
#
# # above line will be different depending on where you saved your data, and your file name
# print(df.groupby(['target', 'time']).agg({'length': 'mean', 'exclaim':'mean', 'point':'mean', 'capital':'mean', 'capital_lower_ration':'mean', 'multiple':'mean'}))


def explore_dataset():
    cvec = CountVectorizer(preprocessor=tweet_cleaner, stop_words=gensim.parsing.preprocessing.STOPWORDS)
    cvec.fit(df.text)
    neg_doc_matrix = cvec.transform(df[df.target == 0].text)
    pos_doc_matrix = cvec.transform(df[df.target == 4].text)
    neg_tf = np.sum(neg_doc_matrix, axis=0)
    pos_tf = np.sum(pos_doc_matrix, axis=0)
    neg = np.squeeze(np.asarray(neg_tf))
    pos = np.squeeze(np.asarray(pos_tf))
    term_freq_df = pd.DataFrame([neg, pos],
                                columns=cvec.get_feature_names()
                                ).transpose()
    term_freq_df.columns = ['negative', 'positive']
    term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False).iloc[:10]
    y_pos = np.arange(50)
    plt.figure(figsize=(12, 10))
    plt.bar(y_pos,
            term_freq_df.sort_values(by='negative', ascending=False)
            ['negative'][:50],
            align='center',
            alpha=0.5)
    plt.xticks(y_pos,
               term_freq_df.sort_values(by='negative', ascending=False)
               ['negative']
               [:50].index,
               rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 50 negative tokens')
    plt.title('Top 50 tokens in negative tweets')
    plt.show()
    y_pos = np.arange(50)
    plt.figure(figsize=(12, 10))
    plt.bar(y_pos,
            term_freq_df.sort_values(by='positive', ascending=False)
            ['positive'][:50],
            align='center',
            alpha=0.5)
    plt.xticks(y_pos,
               term_freq_df.sort_values(by='positive', ascending=False)
               ['positive']
               [:50].index,
               rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 50 positive tokens')
    plt.title('Top 50 tokens in positive tweets')
    plt.show()


def explore_dataset():
    cvec = CountVectorizer(preprocessor=tweet_cleaner, stop_words=gensim.parsing.preprocessing.STOPWORDS)
    cvec.fit(df.text)
    neg_doc_matrix = cvec.transform(df[df.target == 0].text)
    pos_doc_matrix = cvec.transform(df[df.target == 4].text)
    neg_tf = np.sum(neg_doc_matrix, axis=0)
    pos_tf = np.sum(pos_doc_matrix, axis=0)
    neg = np.squeeze(np.asarray(neg_tf))
    pos = np.squeeze(np.asarray(pos_tf))
    term_freq_df = pd.DataFrame([neg, pos],
                                columns=cvec.get_feature_names()
                                ).transpose()
    term_freq_df.columns = ['negative', 'positive']
    term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False).iloc[:10]
    y_pos = np.arange(50)
    plt.figure(figsize=(12, 10))
    plt.bar(y_pos,
            term_freq_df.sort_values(by='negative', ascending=False)
            ['negative'][:50],
            align='center',
            alpha=0.5)
    plt.xticks(y_pos,
               term_freq_df.sort_values(by='negative', ascending=False)
               ['negative']
               [:50].index,
               rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 50 negative tokens')
    plt.title('Top 50 tokens in negative tweets')
    plt.show()
    y_pos = np.arange(50)
    plt.figure(figsize=(12, 10))
    plt.bar(y_pos,
            term_freq_df.sort_values(by='positive', ascending=False)
            ['positive'][:50],
            align='center',
            alpha=0.5)
    plt.xticks(y_pos,
               term_freq_df.sort_values(by='positive', ascending=False)
               ['positive']
               [:50].index,
               rotation='vertical')
    plt.ylabel('Frequency')
    plt.xlabel('Top 50 positive tokens')
    plt.title('Top 50 tokens in positive tweets')
    plt.show()

explore_dataset()


## End of Question 1.

# Train a machine learning model to predict the sentiment of the tweet. Evaluate two
# models\approaches, and tune parameters. One of the models should be based on ‘deep learning’
# approach. Evaluation metrics: accuracy. Present train and test accuracy for the different models and preprocessing combinations.

# X_train, X_test, y_train, y_test = train_test_split(df.drop(labels="target", axis=1), df.target, test_size=0.2, random_state=0)
#
#
pipeline = Pipeline([
    ('features', FeatureUnion([
        # ('exclaim', ExclaimTransformer()),
        ('exclaim', Pipeline([
            ('column_extractor', ColumnExtractor("text")),
            ('exclaim', ExclaimTransformer()),
        ])),
        ('timeOfDay', Pipeline([
            ('column_extractor', ColumnExtractor("date")),
            ('exclaim', TimeOfDayTransformer()),
        ])),
        ('ngram_tf_idf', Pipeline([
            ('column_extractor', ColumnExtractor("text")),
            ('counts', CountVectorizer()),
            ('tf_idf', TfidfTransformer())
        ])),

    ])),

    ('clf', LogisticRegression(n_jobs=-1)),
])

parameters = {
    'features__ngram_tf_idf__counts__max_df': (0.5, 0.75),
    'features__ngram_tf_idf__counts__max_features': (5000, 10000),
    'features__ngram_tf_idf__counts__ngram_range': ((1, 1), (1, 2)),  # unigrams, bigrams ot trigrams
    'features__ngram_tf_idf__counts__preprocessor': (None, tweet_cleaner),
    'clf__penalty': ('l2', None),

}

best_score = 0
best_model = None
is_cnn_best = False


# grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
#
# print("Performing grid search...")
# print("pipeline:", [name for name, _ in pipeline.steps])
# print("parameters:")
# print(parameters)
# # t0 = time()
# grid_search.fit(X_train, y_train)
# # print("done in %0.3fs" % (time() - t0))
# print()
#
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
#
# best_model = grid_search.best_estimator_
# best_score = grid_search.best_score_

### Done with non-neural network model.

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

# Preprocess tweets
df['clean_text'] = df.text.map(tweet_cleaner)

# Split dataset
sentences_train, sentences_test, y_train, y_test = train_test_split(
    df.clean_text.values, df.label.values, test_size=0.25, random_state=1000)

# Fit tokenizer
model_tokenizer = Tokenizer(num_words=5000)
model_tokenizer.fit_on_texts(sentences_train)

# Convert text to sequence
X_train = model_tokenizer.texts_to_sequences(sentences_train)
X_test = model_tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(model_tokenizer.word_index) + 1

maxlen = 200

# Make all tweets max size of 200 tokens. Pad shorter ones with zeros.
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embedding_dim = 100

text_input_layer = layers.Input(shape=(maxlen,))
embedding_layer = layers.Embedding(vocab_size, 50)(text_input_layer)
text_layer = layers.Conv1D(256, 3, activation='relu')(embedding_layer)
text_layer = layers.MaxPooling1D(3)(text_layer)
text_layer = layers.Conv1D(256, 3, activation='relu')(text_layer)
text_layer = layers.MaxPooling1D(3)(text_layer)
text_layer = layers.Conv1D(256, 3, activation='relu')(text_layer)
text_layer = layers.MaxPooling1D(3)(text_layer)
text_layer = layers.GlobalMaxPooling1D()(text_layer)
text_layer = layers.Dense(256, activation='relu')(text_layer)
output_layer = layers.Dense(1, activation='sigmoid')(text_layer)
model = Model(text_input_layer, output_layer)

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=3,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=512)
loss, accuracy = model.evaluate(X_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

if accuracy > best_score:
    best_score = accuracy
    best_model = model
    is_cnn_best = True


def predict_on_new_tweets(model, is_cnn_best):
    colnames = ['tweet_id', 'date', 'query', 'user', 'text']
    df = pd.read_csv("Test.csv", encoding='latin-1', names=colnames, header=None, engine='python',
                     error_bad_lines=False)

    if is_cnn_best:
        X = df.text.map(tweet_cleaner).values
        X = model_tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, padding='post', maxlen=maxlen)
    else:
        X = df.text

    test_pred = model.predict(X)
    i = 0
    finalans = ["ID,Sentiment"]
    for val in test_pred:
        finalans.append(str(i) + "," + str(int(round(val[0]))))
        i += 1
    print(finalans)

    with open("out1.csv", "w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(finalans)