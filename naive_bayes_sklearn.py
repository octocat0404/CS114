import pandas as pd
from pathlib import Path

train = pd.read_csv(Path('./data/train.csv'))
test  = pd.read_csv(Path('./data/test.csv'))
val   = pd.read_csv(Path('./data/valid.csv'))

# Chia dữ liệu theo từng lớp
class_0 = train[train['sentiment'] == 0]
class_2 = train[train['sentiment'] == 2]
class_1 = train[train['sentiment'] == 1]  # Lớp thiểu số

# Random sample từ lớp 0 và 2 để số lượng bằng lớp 1
class_0_under = class_0.sample(len(class_1), random_state=42)
class_2_under = class_2.sample(len(class_1), random_state=42)

# Kết hợp lại thành tập dữ liệu mới
balanced_data = pd.concat([class_0_under, class_2_under, class_1])
print(balanced_data['sentiment'].value_counts())

X = balanced_data['sentence']
y = balanced_data['sentiment']

import re
# Tách các stop words
# Tạo danh sách stop words trong tiếng Việt
stop_words_vi = set(["là", "và", "một", "có", "những", "cho", "được", "tại", "với"])

def remove_stopwords_vi(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words_vi]
    return ' '.join(filtered_words)

# Xử lý khoảng trắng trong câu
def clean_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

from pyvi import ViTokenizer
# Tạo bản sao để tránh thay đổi dữ liệu gốc
df_train = train.copy()

# Áp dụng vectorization
def preprocess_text(text):
    text = clean_whitespace(text)
    text = remove_stopwords_vi(text)
    return ViTokenizer.tokenize(text)

df_train['sentence'] = df_train['sentence'].apply(preprocess_text)
val['sentence'] = val['sentence'].apply(preprocess_text)
test['sentence'] = test['sentence'].apply(preprocess_text)

X_train = df_train['sentence']
y_train = df_train['sentiment']
X_val = val['sentence']
y_val = val['sentiment']
X_test = test['sentence']
y_test = test['sentiment']

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from pyvi import ViTokenizer

#Xây dựng pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer=ViTokenizer.tokenize,  # Sử dụng pyvi để tách từ
        ngram_range=(1, 2),  # Xét cả unigram và bigram
        max_features=5000  # Giới hạn số features
    )),
    ('nb', MultinomialNB(
        alpha=1.0,  # Laplace smoothing
        fit_prior=True  # Tính prior probability từ dữ liệu
    ))
])
#Huấn luyện mô hình
pipeline.fit(X_train, y_train)
#Đánh giá mô hình
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib 

joblib.dump(pipeline, './model/naive_bayes_sklearn.sav', compress=0)
joblib.dump(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose(), './eval/naive_bayes_sklearn.sav', compress=0)