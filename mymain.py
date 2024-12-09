import numpy as np
import polars as pl

# scipy
# sklearn
# nltk, gensim
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
# glmnet, XGBoost and CatBoost

seed = 3794631
np.random.seed(seed)

train = pl.read_csv("train.csv")
test = pl.read_csv("test.csv")
Y_test = pl.read_csv("test_y.csv")

# 1. Build a Binary Classification Model

# The first objective is to construct a binary classification model to predict the sentiment of a movie review.

# The evaluation metric for this project is the Area Under the Curve (AUC) on the test data. Your goal is to achieve an AUC score of at least 0.986 across all five test data splits.

# logistic regression



#For the interpretability analysis, a binary classification model trained on split1 should have been saved online, so access to train.csv is no longer required. If the ten randomly selected test samples are saved online, access to the original test.csv with OpenAI embeddings also is no longer required. But you may need online access to test data with the (non-OpenAI) embeddings you selected.

#we also need to save the linear transform matrix W, which is used to align non-OpenAI embeddings and OpenAI embeddings online
#As with the ten random samples, you may choose to save W or not. The main advantage of saving is faster execution the next time the code is run, because you won't need to regenerate the samples or W.


test.write_csv("mysubmission.csv")


# 2. Interpretability Analysis
# Using split 1 and the corresponding trained model, implement an interpretability approach to identify which parts of each review have an impact on the sentiment prediction. Apply your method to 5 randomly selected positive reviews and 5 randomly selected negative reviews from the split 1 test data.

# Set a random seed before selecting these 10 reviews (the seed does not need to relate to students’ UINs).

# Provide visualizations (such as highlighted text) that show the key parts of a review contributing to the sentiment prediction. Discuss the effectiveness and limitations of the interpretability approach you chose.

