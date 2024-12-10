import numpy as np
import pandas as pd
import polars as pl
import os
import time
# import nltk
from glmnet import LogitNet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing 
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import pickle
from scipy.stats import uniform
# tensorflow
# pytorch
# BeautifulSoup
# re
# collections
# string
# torchtext
# spacy
# tensorflow
# eli5
# Fasttext
# keras

seed = 3794631
np.random.seed(seed)

train = pl.read_csv("train.csv")
# print(train.shape) #id, sentiment, review + 1536 embedding columns

Y_train = train["sentiment"]
X_train_IDs = train["id"]
X_train = train.drop(["id", "sentiment", "review"])

scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
  
test = pl.read_csv("test.csv")
X_test_IDs = test["id"]
print(X_test_IDs)
X_test = test.drop(["id", "review"])
X_test_scaled = scaler.transform(X_test)

test_y = pl.read_csv("test_y.csv")
Y_test = test_y.drop("id")

# 1. Build a Binary Classification Model

# The first objective is to construct a binary classification model to predict the sentiment of a movie review.

# The evaluation metric for this project is the Area Under the Curve (AUC) on the test data. Your goal is to achieve an AUC score of at least 0.986 across all five test data splits.

# logistic regression

# logreg = LogisticRegression(random_state=356,solver='saga')

# logreg_model = logreg.fit(X_scaled, Y_train)

# mypred = logreg_model.predict_proba(X_test)

# AUC = roc_auc_score(Y_test, mypred[:,1])

# # print(mypred[:,1])
# print(AUC) #0.9838018188257659

# # save model
# with open("logreg.pkl", "wb") as f:
#     pickle.dump(logreg_model, f)

# load model
# https://github.com/emeraldrains/CS598-Project-3/raw/refs/heads/main/logreg.pkl
with open('logreg.pkl', 'rb') as f:
    logreg_loaded = pickle.load(f)

mypred = logreg_loaded.predict_proba(X_test_scaled)
AUC = roc_auc_score(Y_test, mypred[:,1])
# print(mypred[:,1])
print(AUC) #0.9838018188257659


start = time.time()

# logit = LogitNet()
# logit = logit.fit(X_scaled, Y_train)

# # save model
# with open("logitnet.pkl", "wb") as f:
#     pickle.dump(logit, f)

# load model
# https://github.com/
with open('logitnet.pkl', 'rb') as f:
    logitnet_loaded = pickle.load(f)

mypred = logitnet_loaded.predict_proba(X_test_scaled)
AUC = roc_auc_score(Y_test, mypred[:,1])

end = time.time()
print(f"it takes {end-start} seconds")


test_res = pl.DataFrame({
    "id": X_test_IDs,
    "prob": mypred[:,1]
})

test_res.write_csv("mysubmission.csv")


# print(mypred[:,1])
print(AUC) #0.9862533758637505


# create mysubmission

#Apply your method to 5 randomly selected positive reviews and 5 randomly selected negative reviews from the split 1 test data.

# random_pos_rows = test_y.filter(pl.col("sentiment") == 1).sample(n=5)
# random_neg_rows = test_y.filter(pl.col("sentiment") == 0).sample(n=5)

# # print(random_pos_rows)
# pos_test = (test.join(random_pos_rows, on="id"))

# # print(random_neg_rows)
# neg_test = (test.join(random_neg_rows, on="id"))


# test_set = pl.concat([pos_test,neg_test])
# test_ids = test_set["id"]
# Y_test_subset = test_set["sentiment"]
# test_set = test_set.drop(["id", "sentiment", "review"])
# test_set = scaler.transform(test_set)

# mypred_test = logitnet_loaded.predict_proba(test_set)

# print(test_ids.join(mypred_test[:,1])) # probability of being in class 1
# test_res = pl.DataFrame({
#     "id": test_ids,
#     "prob": mypred_test[:,1]
# })

# test_res.write_csv("mysubmission.csv")



#For the interpretability analysis, a binary classification model trained on split1 should have been saved online, so access to train.csv is no longer required. If the ten randomly selected test samples are saved online, access to the original test.csv with OpenAI embeddings also is no longer required. But you may need online access to test data with the (non-OpenAI) embeddings you selected.

#we also need to save the linear transform matrix W, which is used to align non-OpenAI embeddings and OpenAI embeddings online
#As with the ten random samples, you may choose to save W or not. The main advantage of saving is faster execution the next time the code is run, because you won't need to regenerate the samples or W.




# values less than 0.5 indicate a negative review and values greater than 0.5 indicate a positive review.



# id,     prob
# 47604,  0.940001011154441
# 36450,  0.584891891011812
# 30088,  0.499236341444505
# 18416,  0.0068778600913503






# 2. Interpretability Analysis
# Using split 1 and the corresponding trained model, implement an interpretability approach to identify which parts of each review have an impact on the sentiment prediction. Apply your method to 5 randomly selected positive reviews and 5 randomly selected negative reviews from the split 1 test data.

# Set a random seed before selecting these 10 reviews (the seed does not need to relate to students’ UINs).

# Provide visualizations (such as highlighted text) that show the key parts of a review contributing to the sentiment prediction. Discuss the effectiveness and limitations of the interpretability approach you chose.



