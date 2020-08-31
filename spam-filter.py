#   _____                                 ______   _   _   _
#  / ____|                               |  ____| (_) | | | |
# | (___    _ __     __ _   _ __ ___     | |__     _  | | | |_    ___   _ __
#  \___ \  | '_ \   / _` | | '_ ` _ \    |  __|   | | | | | __|  / _ \ | '__|
#  ____) | | |_) | | (_| | | | | | | |   | |      | | | | | |_  |  __/ | |
# |_____/  | .__/   \__,_| |_| |_| |_|   |_|      |_| |_|  \__|  \___| |_|
#          | |
#          |_|

# Author: Vesna Zupanc

import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# -----------------------------------
# IMPORTING AND EXPLORING DATASET
# -----------------------------------

data = pd.read_csv(r"C:\Users\Vesna\PycharmProjects\spam-predictor\spam_ham_dataset.csv")

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5171 entries, 0 to 5170
# Data columns (total 4 columns):
#   #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#   0   Unnamed: 0  5171 non-null   int64
#   1   label       5171 non-null   object
#   2   text        5171 non-null   object
#   3   label_num   5171 non-null   int64
# dtypes: int64(2), object(2)
# memory usage: 161.7+ KB

# Labels:
#   - ham = 0
#   - spam = 1

# All we need is text and label_num columns
df = data[['text', 'label_num']].rename(columns={'label_num': 'label'})


def save_table(table, name, tablefmt="grid"):
    with open(f"C:/Users/Vesna/PycharmProjects/spam-predictor/summary/tables/{name}.md", "w") as f:
        print(table.to_markdown(tablefmt=tablefmt), file=f)
    f.close()


save_table(df.head(2), 'table1')

# How is our data distributed?


fig, ax = plt.subplots()
df_temp = df.groupby("label").count().reset_index()
ax.bar(df_temp.label.astype(str), df_temp.text)
fig.savefig('C:/Users/Vesna/PycharmProjects/spam-predictor/summary/plots/spam_ham_dist.png')  # save the figure to file
plt.close(fig)

fig, ax = plt.subplots()
df_temp['perc'] = df_temp.text / df_temp.text.sum()
ax.pie(df_temp.perc, labels=['ham', 'spam'], autopct='%1.1f%%')
plt.ylabel('')
fig.savefig(
    'C:/Users/Vesna/PycharmProjects/spam-predictor/summary/plots/spam_ham_dist_perc.png')  # save the figure to file
plt.close(fig)

# -----------------------------------
# TEXT ANALYSIS
# -----------------------------------

# removing stopwords


stopwords_set = set(stopwords.words('english'))
stopwords_set.add('subject')


def preprocessing_text(x):
    x = x.lower()
    x = re.sub(r'\d+', '', x)
    x = re.sub(r'[^\w\s]', '', x)
    x = x.strip()
    x = ' '.join([word for word in word_tokenize(x) if word not in stopwords_set])
    return x


df['text'] = df['text'].apply(lambda x: preprocessing_text(x))

save_table(df.head(2), 'table_cleared')

# HAM PLOT
fig, ax = plt.subplots(figsize=[6.4, 6.2])
df_temp = pd.DataFrame.from_dict(Counter(" ".join(df[df.label == 0]['text']).split()).most_common(10)).rename(
    columns={0: "besede", 1: "frekvenca"})
ax.bar(df_temp.besede, df_temp.frekvenca)
plt.xticks(rotation=30)
fig.savefig('C:/Users/Vesna/PycharmProjects/spam-predictor/summary/plots/ham_words.png')
plt.close(fig)

# SPAM PLOT
fig, ax = plt.subplots(figsize=[6.4, 6.2])
df_temp = pd.DataFrame.from_dict(Counter(" ".join(df[df.label == 1]['text']).split()).most_common(10)).rename(
    columns={0: "besede", 1: "frekvenca"})
ax.bar(df_temp.besede, df_temp.frekvenca)
plt.xticks(rotation=30)
fig.savefig('C:/Users/Vesna/PycharmProjects/spam-predictor/summary/plots/spam_words.png')
plt.close(fig)

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X = df.text
y = df.label


# Splitting data to train and test
def print_dist(x):
    val_count = x.value_counts()
    val_all = val_count.sum()
    print(f"Percentage of values:"
          f"\n\t * 0: {round(val_count[0] / val_all * 100, 2)}%"
          f"\n\t * 1: {round(val_count[1] / val_all * 100, 2)}%")
    return None


print_dist(y)
# Percentage of values:
# 	 * 0: 71.01%
# 	 * 1: 28.99%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print_dist(y_train)
# Percentage of values:
# 	 * 0: 71.3%
# 	 * 1: 28.7%
# print_dist(y_test)
# Percentage of values:
# 	 * 0: 69.86%
# 	 * 1: 30.14%


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# print_dist(y_train)
# Percentage of values:
# 	 * 0: 71.01%
# 	 * 1: 28.99%
# print_dist(y_test)
# Percentage of values:
# 	 * 0: 71.01%
# 	 * 1: 28.99%

# Same distibution! :)

# -----------------------------------
# MAKING PIPELINES
# -----------------------------------


# CountVectorization and MultinomialNaiveBayes
pipeline1 = Pipeline([
    ('counts', CountVectorizer(ngram_range=(1, 2))),
    ('nb', MultinomialNB())
])

# CountVectoriztion and ComplementNaiveBayes
pipeline2 = Pipeline([
    ('counts', CountVectorizer(ngram_range=(1, 2))),
    ('cnb', ComplementNB())
])

# TF-IDF and LogisticRegression
pipeline3 = Pipeline([
    ('tfid', TfidfVectorizer()),
    ('lr', LogisticRegression())
])

# TF-IDF and SVC
pipeline4 = Pipeline([
    ('tfid', TfidfVectorizer()),
    ('svc', SVC())
])

# Fitting models


pipeline1.fit(X_train, y_train)

pipeline2.fit(X_train, y_train)

pipeline3.fit(X_train, y_train)

pipeline4.fit(X_train, y_train)

# Confusin matrix
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[6.4, 6.4])
plot_confusion_matrix(pipeline1, X_test, y_test, cmap=plt.cm.Blues, ax=axs[0, 0])
axs[0, 0].set_title('MultinomalNB')
plot_confusion_matrix(pipeline2, X_test, y_test, cmap=plt.cm.Blues, ax=axs[0, 1])
axs[0, 1].set_title('ComplementNB')
plot_confusion_matrix(pipeline3, X_test, y_test, cmap=plt.cm.Blues, ax=axs[1, 0])
axs[1, 0].set_title('LogisticRegression')
plot_confusion_matrix(pipeline4, X_test, y_test, cmap=plt.cm.Blues, ax=axs[1, 1])
axs[1, 1].set_title('SVC')
fig.savefig('C:/Users/Vesna/PycharmProjects/spam-predictor/summary/plots/conf_mtr.png')
plt.close(fig)

# Confusin matrix normalized
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[6.4, 6.4])
plot_confusion_matrix(pipeline1, X_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=axs[0, 0])
axs[0, 0].set_title('MultinomalNB')
plot_confusion_matrix(pipeline2, X_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=axs[0, 1])
axs[0, 1].set_title('ComplementNB')
plot_confusion_matrix(pipeline3, X_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=axs[1, 0])
axs[1, 0].set_title('LogisticRegression')
plot_confusion_matrix(pipeline4, X_test, y_test, cmap=plt.cm.Blues, normalize='true', ax=axs[1, 1])
axs[1, 1].set_title('SVC')
fig.savefig('C:/Users/Vesna/PycharmProjects/spam-predictor/summary/plots/conf_mtr_norm.png')
plt.close(fig)

# Precision, recall, f
y1 = pipeline1.predict(X_test)
y2 = pipeline2.predict(X_test)
y3 = pipeline3.predict(X_test)
y4 = pipeline4.predict(X_test)

precision = [precision_score(y_test, y_pred) for y_pred in [y1, y2, y3, y4]]
recall = [recall_score(y_test, y_pred) for y_pred in [y1, y2, y3, y4]]
f1 = [f1_score(y_test, y_pred) for y_pred in [y1, y2, y3, y4]]

df_measures = pd.DataFrame(
    dict(zip(['Model', 'Preciznost', 'Priklic', 'F-mera'], [['NB', 'CNB', 'LR', 'SVC'], precision, recall, f1])))
df_measures = df_measures.set_index('Model')
df_measures = df_measures.round(4)
save_table(df_measures, 'table_mesures', tablefmt="grid")


# -----------------------------------
# k-cross validation
# -----------------------------------


def k_cross_valid(pipeline):
    k_fold = KFold(n_splits=6)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold.split(df):
        train_text = df.iloc[train_indices]['text'].values
        train_y = df.iloc[train_indices]['label'].values

        test_text = df.iloc[test_indices]['text'].values
        test_y = df.iloc[test_indices]['label'].values

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        score = f1_score(test_y, predictions)
        scores.append(score)
    return sum(scores) / len(scores)


scores = [k_cross_valid(pipeline_i) for pipeline_i in [pipeline1, pipeline2, pipeline3, pipeline4]]
df_measures_2 = pd.DataFrame(dict(zip(['Model', 'Score'], [['NB', 'CNB', 'LR', 'SVC'], scores])))
df_measures_2 = df_measures_2.set_index('Model')
df_measures_2 = df_measures_2.round(4)
save_table(df_measures_2, 'table_mesures_KFold', tablefmt="grid")
