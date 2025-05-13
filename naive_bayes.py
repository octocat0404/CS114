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
df_train.head

val['sentence'] = val['sentence'].apply(preprocess_text)
val.head

test['sentence'] = test['sentence'].apply(preprocess_text)
test.head

X_train = df_train['sentence']
y_train = df_train['sentiment']
X_val = val['sentence']
y_val = val['sentiment']
X_test = test['sentence']
y_test = test['sentiment']

import numpy as np
from collections import defaultdict
import math
from pyvi import ViTokenizer


def int_defaultdict():
    return defaultdict(int)

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Khởi tạo mô hình
        alpha: tham số làm mịn Laplace (mặc định là 1.0)
        """
        self.alpha = alpha
        self.class_word_counts = defaultdict(int_defaultdict)
        self.class_counts = defaultdict(int)
        self.class_total_words = defaultdict(int)
        self.vocab = set()
        self.classes = None
    
    def fit(self, X, y):
        """
        Huấn luyện mô hình
        X: danh sách các văn bản đã được tiền xử lý
        y: nhãn tương ứng
        """
        self.classes = np.unique(y)
        
        for text, label in zip(X, y):
            self.class_counts[label] += 1
            words = text.split()
            for word in words:
                self.class_word_counts[label][word] += 1
                self.class_total_words[label] += 1
                self.vocab.add(word)
        
        self.vocab_size = len(self.vocab)
    
    def _calculate_log_prob(self, word, label):
        """
        Tính log xác suất P(word|label) với làm mịn Laplace
        """
        word_count = self.class_word_counts[label].get(word, 0)
        total_words = self.class_total_words[label]
        return math.log((word_count + self.alpha) / (total_words + self.alpha * self.vocab_size))
    
    def predict(self, X):
        """
        Dự đoán nhãn cho các văn bản đầu vào
        X: danh sách các văn bản cần dự đoán
        """
        predictions = []
        for text in X:
            words = text.split()
            best_score = -float('inf')
            best_class = None
            
            for label in self.classes:
                # Tính log prior P(label)
                log_prob = math.log(self.class_counts[label] / sum(self.class_counts.values()))
                
                # Tính log likelihood P(words|label)
                for word in words:
                    log_prob += self._calculate_log_prob(word, label)
                
                # Chọn lớp có xác suất cao nhất
                if log_prob > best_score:
                    best_score = log_prob
                    best_class = label
            
            predictions.append(best_class)
        
        return predictions
    
    def score(self, X, y):
        """
        Tính độ chính xác trên tập kiểm tra
        """
        y_pred = self.predict(X)
        return np.mean(np.array(y_pred) == np.array(y))
    
from sklearn.metrics import classification_report

#Huấn luyện mô hình
mnb = MultinomialNaiveBayes(alpha = 0.1)
mnb.fit(X_train, y_train)

#Đánh giá mô hình
y_pred = mnb.predict(X_test)
accuracy = mnb.score(X_test, y_test)
print("Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib 

joblib.dump(mnb, './model/naive_bayes.sav', compress=0)
