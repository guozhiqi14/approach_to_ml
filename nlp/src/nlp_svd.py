import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string


def clean_text(s):
    """
    This function cleans the text a bit
    :param s: string
    :return: cleaned string
    """
    # split by all whitespaces
    s = s.split()

    # join tokens by single space
    # why we do this?
    # this will remove all kinds of weird space
    # "hi. how are you" becomes
    # "hi. how are you"
    s = " ".join(s)

    # remove all punctuations using regex and string module
    s = re.sub(f'[{re.escape(string.punctuation)}]', '', s)
    # you can add more cleaning here if you want
    # and then return the cleaned string
    return s


# create a corpus of sentences
# we read only 10k samples from training data
# for this example
# IMDB dataset download: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download
corpus = pd.read_csv("../input/IMDB_Dataset.csv", nrows=10000)
corpus.loc[:, "review"] = corpus.review.apply(clean_text)
corpus = corpus.review.values

# initialize TfidfVectorizer with word_tokenize from nltk
# as the tokenizer
tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)

# fit the vectorizer on corpus
tfv.fit(corpus)

# transform the corpus using tfidf
corpus_transformed = tfv.transform(corpus)

# initialize SVD with 10 components
svd = decomposition.TruncatedSVD(n_components=10)

# fit SVD
corpus_svd = svd.fit(corpus_transformed)

# choose first sample and create a dictionary
# of feature names and their scores from svd
# you can change the sample_index variable to
# get dictionary for any other sample
sample_index = 0
# feature_scores = dict(zip(tfv.get_feature_names(), corpus_svd.components_[sample_index]))

# once we have the dictionary, we can now
# sort it in decreasing order and get the
# top N topics
N = 5
"""
代码尝试为每个样本提取特征分数，并按分数排序以识别最重要的特征。
tfv.get_feature_names() 获取特征名称（词汇表中的词）。
corpus_svd.components_ 包含 SVD 分解后的特征权重。
sorted(feature_scores, key=feature_scores.get, reverse=True)[:N] 获取前 N 个最重要的特征。
"""
for sample_index in range(5):
    feature_scores = dict(zip(tfv.get_feature_names(), corpus_svd.components_[sample_index]))
    print(sorted(feature_scores, key=feature_scores.get, reverse=True)[:N])

