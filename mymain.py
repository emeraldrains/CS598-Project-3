import numpy as np
import pandas as pd
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


# 1. Build a Binary Classification Model

# The first objective is to construct a binary classification model to predict the sentiment of a movie review.

# The evaluation metric for this project is the Area Under the Curve (AUC) on the test data. Your goal is to achieve an AUC score of at least 0.986 across all five test data splits.

# logistic regression

2. Interpretability Analysis
Using split 1 and the corresponding trained model, implement an interpretability approach to identify which parts of each review have an impact on the sentiment prediction. Apply your method to 5 randomly selected positive reviews and 5 randomly selected negative reviews from the split 1 test data.

Set a random seed before selecting these 10 reviews (the seed does not need to relate to students’ UINs).

Provide visualizations (such as highlighted text) that show the key parts of a review contributing to the sentiment prediction. Discuss the effectiveness and limitations of the interpretability approach you chose.

