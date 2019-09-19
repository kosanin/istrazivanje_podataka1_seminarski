from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import SnowballStemmer

from  params import mlp_params, svm_params, log_params
import pandas as pd
import pickle
import nltk
import re
import eda

TOP_K_FEATURES  = 20000

NEG_REGEX       = re.compile(r"^(\w*?n't|no(t)?|never$)", re.I)
WORD_REGEX      = re.compile(r"^[a-zA-Z_']+$", re.I)
STEMMER         = SnowballStemmer('english')
STOP_WORDS      = frozenset(ENGLISH_STOP_WORDS.union(['movie', 'film']).difference(['not', 'never', 'no']))
ALGS_METRICS     = {}

models = [
    ('MNB', MultinomialNB(), None),
    ('LogReg', LogisticRegression(), log_params),
    ('SVM', LinearSVC(), svm_params),
#    ('MLP', MLPClassifier(), mlp_params),
    ('DT', DecisionTreeClassifier(), None)
]


def preprocess_raw_text(raw_review):
    """
    negates appropriate words
    removes stop words
    performs stemming
    :param raw_review:  raw text
    :return: list od tokens
    """
    tokens = nltk.word_tokenize(text=raw_review, language='english')
    words = []
    negation_flag = False

    for token in tokens:
        if WORD_REGEX.search(token):
            word = "not_" + token if negation_flag else token
            if NEG_REGEX.search(token) is not None:
                negation_flag = True

            if word not in STOP_WORDS:
                words.append(STEMMER.stem(word))

    return words

def term_matrix(df, term_count_mode='bow'):


    """
    makes term matrix
    finds K best attributes
    writes vectorizer and selecotr in file for transforming new data
    return term matrix and labels

    :param df: trening skup
    :param term_count_mode: bow or tfidf
    :return: term matrix, labels
    """
    if term_count_mode == "tfidf":
        vec = TfidfVectorizer(tokenizer=preprocess_raw_text,
                              ngram_range=(1, 2))
    else:
        vec = CountVectorizer(tokenizer=preprocess_raw_text,
                              ngram_range=(1, 2))

    X = vec.fit_transform(df['text'])
    y = df['sentiment']
    selector = SelectKBest(chi2, k=min(TOP_K_FEATURES, X.shape[1]))
    X_new = selector.fit_transform(X, y)

    write_pickle(vec, selector, term_count_mode + "_vect_and_sel.pickle")

    return X_new, y


def write_pickle(X, y, path):

    with open(path, "wb") as f:
        pickle.dump((X, y), f)


def read_pickle(path):

    with open(path, "rb") as f:
        (X, y) = pickle.load(f)
        return X, y


def train_models(X_train, y_train, text_repr):

    """
    trains each model in models list
    for MLP and SVM performs grid search for hyperparameters
    saves trained models in ./models/model_name.pickle


    :param X_train: training data
    :param y_train: labels of training data
    :param text_repr: tfidf or bow
    """

    for name, estimator, params in models:
        if params is not None:
            clf = GridSearchCV(estimator=estimator,
                               param_grid=params,
                               scoring='accuracy',
                               n_jobs=2,
                               refit=True,
                               cv=5)
        else:
            clf = estimator

        clf.fit(X_train, y_train)
        with open("models/" + text_repr + "_" + name + ".pickle", "wb") as f:
            pickle.dump(clf, f)


def model_metrics(model_name, y_test, y_pred):
    """

    :param model_name:
    :param y_test:
    :param y_pred:
    :return:
    """
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    ALGS_METRICS[model_name] = { 'f1' : f1,
                                'acc': acc,
                                'auc': auc}

def write_metrics(text_repr):
    """
    writes algorithms metrics in text_repr_metrics.csv
    :param text_repr: bow or tfidf
    """
    df = pd.DataFrame(ALGS_METRICS).T
    df.to_csv(text_repr +"_metrics.csv")
    ALGS_METRICS.clear()


def eval_models(X_test, y_test, text_repr):
    """
    calculate performance metrics for all models

    :param X_test: test data set
    :param y_test: labels for test data set
    :param text_repr: bow or tfidf
    """
    for name, _, _ in models:
        with open('models/' + text_repr + "_" + name + '.pickle', "rb") as f:
            clf = pickle.load(f)
            model_metrics(name, y_test, clf.predict(X_test))
    write_metrics(text_repr)


def train_eval_all():
    """
    Trenira sve modele koji se zatim cuvaju u odgovarajuci fajl
    Nakon treniranja svih modela, modeli se evaluiraju

    problem ove funkcije je da nepotrebno cita podatke
    (umesto direktne evaluacije)
    ali koristi se samo jednom pa nije potrebno refaktorisanje
    """
    text_repr = ['bow']
    for rep in text_repr:
        X, y = read_pickle('data/' + rep + ".pickle")
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)
        train_models(X_train, y_train, rep)
        eval_models(X_test, y_test, rep)


def predict_sentiment(text, labels, text_repr, algorithm):
    """
    transforms text into text_repr and classifies with algorithm algorithm.
    :param text: list of document(s) to be classified
    :param labels: list of true labels
    :param text_repr: bow or tfidf
    :param algorithm: SVM, DT, MNB, LogReg (CASE SENSITIVE!)
    :return: dataframe with columns true_class, predicted
    """
    with open('data/' + text_repr + '_vect_and_sel.pickle', 'rb') as vs:
        vec, sel = pickle.load(vs)
        X = sel.transform(vec.transform(text))

        with open('models/' + text_repr + '_' + algorithm + '.pickle', "rb") as f:
            clf = pickle.load(f)
            predicted = clf.predict(X)
            return pd.DataFrame({'true_class' : labels,
                                 'predicted'  : predicted})


def main():

   # df = pd.read_csv('original_data/train.csv').head(100)
   # print(predict_sentiment(df['text'], df['sentiment'], 'bow', 'SVM'))
    x =predict_sentiment(text=pd.read_csv('test_sample5.csv')['text'],
                labels=[1, 0, 0, 0, 1, 1],
                text_repr='tfidf',
                algorithm='SVM')

    print(x)


if __name__ == '__main__':
    main()