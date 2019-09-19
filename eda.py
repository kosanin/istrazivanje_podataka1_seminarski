import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

MIN_OCCS        = 100
FIRST_N_OCCS    = 30


def make_sample5(df, path):
    """

    :param df: dataframe
    writes 5 shortest movie reviews in sample5.csv
    """

    reviews_lens = df['text'].map(len)

    df = df.merge(reviews_lens,
                  left_index=True,
                  right_index=True,
                  how='inner')

    df.sort_values(by='text_y', inplace=True)

    df.rename(index = str,
              columns={'text_x' : 'text'},
              inplace=True)

    df.head().to_csv(path_or_buf=path,
                     columns=['text', 'sentiment'],
                     index=False)

def class_distribution_plot(df):
    """

    :param df: dataframe containing dataset
    plots bar plot for class (sentiment) distribution
    """
    df.groupby('sentiment').count().plot(kind='barh')
    plt.show()

def word_occs_plot(df):

    """

    :param df: dataframe
    plots FIRST_N_OCCS most occuring words
    """

    vect = CountVectorizer(stop_words='english')
    cv = vect.fit_transform(df['text'])

    word_occs = cv.sum(axis=0)
    word_freq = [(word, word_occs[0, idx]) for word, idx in vect.vocabulary_.items()]
    word_freq = sorted((filter(lambda x: x[1] > MIN_OCCS, word_freq)),
                       key=lambda x: x[1],
                       reverse=True)[:FIRST_N_OCCS]

    df1 = pd.DataFrame(word_freq, columns=['text', 'count'])

    df1.plot.bar(x='text', y='count')
    plt.show()

def plot_conf_matrix(y_true, y_pred):
    """
    TN        FP
    FN        TP


    :param y_true: labels of test data
    :param y_pred: predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, columns=['positive', 'negative'], index=['positive', 'negative'])

    fig = plt.figure()
    sn.heatmap(cm_df, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()

def plot_conf_matrix_df(df):
    """
             positive negative
    positive
    negative

        0   1
    0
    1
    :param df: dataframe with true_class and predicted columns
    """
    cm = confusion_matrix(df['true_class'].to_list(), df['predicted'].to_list())
    cm_df = pd.DataFrame(cm, columns=['positive', 'negative'], index=['positive', 'negative'])

    plt.figure()
    sn.heatmap(cm_df, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()
