# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 04:33:09 2024

@author: liwei
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Embedding, SpatialDropout1D
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')

import numpy as np
import pandas as pd
import re
import jsonlines

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression



def vectorise(texts, w2v_model, max_length):
    vector_size = w2v_model.vector_size
    texts_vec = []

    words = set(w2v_model.wv.index_to_key)

    for text in texts:
        sentence_vectors = []

        # 遍历句子中的每个词，最多到 max_length
        for word in text[:max_length]:
            if word in words:
                # 如果词在模型的词汇表中，添加它的向量
                sentence_vectors.append(w2v_model.wv[word])
            else:
                # 否则，添加零向量
                sentence_vectors.append(np.zeros(vector_size))

        # 如果句子长度小于 max_length，用零向量填充剩余的部分
        for _ in range(max_length - len(sentence_vectors)):
            sentence_vectors.append(np.zeros(vector_size))

        # 将句子的词向量列表添加到结果列表中
        texts_vec.append(sentence_vectors)

    # 将列表转换为三维 NumPy 数组
    # 结果形状为 (句子数量, max_length, vector_size)
    return np.array(texts_vec)

def process_strings(strings):
    returned = []
    for case in strings:
        if not isinstance(case, str):  # 检查case是否为字符串
            case = str(case)  # 不是字符串时转换为字符串
        case = re.sub(r'\[[0-9, ]*\]', '', case)
        case = re.sub(r'^...', '... ', case)
        case = word_tokenize(case.lower())
        if not case:
            case = [' ']
        returned.append(case)
    return returned

def process_names(sectionNames):
    returned = []
    for case in sectionNames:
        print(case)
        case = case.lower()
        case = re.sub(r'^[0-9.]{2,}', '', case)
        returned.append(case)
    return returned

def train(model, x_train, y_train):
    model = model.fit(x_train, y_train)

def predict(model, x_test):
    return model.predict(x_test)

def evaluate(y_test, y_pred):
    score = f1_score(y_test, y_pred, average='macro')
    print('f1 score = {}'.format(score))
    print('accuracy = %s' % accuracy_score(y_test, y_pred))

def parse_label2index(label):
    index = []
    for i in range(len(label)):
        if label[i] == "background":
            index.append(0)
        elif label[i] == "method":
            index.append(1)
        else: # label[i] == "result"
            index.append(2)
    return index

def parse_index2label(index):
    label = []
    for i in range(len(index)):
        if index[i] == 0:
            label.append("background")
        elif index[i] == 1:
            label.append("method")
        else: # index[i] == 2
            label.append("comparison")
    return label

def create_lstm_model(n1,n2):
    model = Sequential()
    #model.add(LSTM(128, input_shape=(33, 100), dropout=0.3, recurrent_dropout=0.3))
    model.add(Bidirectional(LSTM(128,dropout=0.3, recurrent_dropout=0.3), input_shape=(2*(n1+n2), 100)))
    # 其他层...
    model.add(Dense(32, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
class F1ScoreCallback(Callback):
    def __init__(self, train_data, val_data):
        super(F1ScoreCallback, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_f1_scores = []
        self.val_f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
    
    # Predict classes with the model
        y_train_pred = np.argmax(self.model.predict(X_train), axis=-1)
        y_val_pred = np.argmax(self.model.predict(X_val), axis=-1)

    # Calculate F1 scores
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
    
    # Append the F1 scores to the lists
        self.train_f1_scores.append(train_f1)
        self.val_f1_scores.append(val_f1)
    
    # Print the scores
        print(f'Epoch {epoch+1} - train F1: {train_f1:.4f}, val F1: {val_f1:.4f}')

    def on_train_end(self, logs=None):
        # 绘制F1分数变化趋势图
        plt.plot(self.train_f1_scores, label='Train F1')
        plt.plot(self.val_f1_scores, label='Validation F1')
        plt.title('F1 Score Trend')
        plt.ylabel('F1 Score')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

def main():
    word2vec_model = Word2Vec.load('C:/Users/liwei/Desktop/CS4248/word2vec_model.bin')
    
    sectionNames, strings, labels, label_confidence, isKeyCitation = [], [], [], [], []
    with jsonlines.open('C:/Users/liwei/Desktop/CS4248/train.jsonl') as f:
        for line in f.iter():
            sectionNames.append(line['sectionName'])  #handle NaN?
            strings.append(re.sub(r'\n', '. ', line['string']))
            labels.append(line['label'])
            # label_confidence.append(line['label_confidence']) #use?
            # isKeyCitation.append(line['isKeyCitation']) #use?
    strings = process_strings(strings)
    sectionNames = process_strings(sectionNames)
    y_train = parse_label2index(labels)
    
    n1=33
    n2=2
    #x_train = vectorise(strings, word2vec_model,n1)
    X_train = np.concatenate([vectorise(strings, word2vec_model,n1), vectorise(sectionNames, word2vec_model,n2)], axis=1)
    y_train = parse_label2index(labels)
    y_train = np.array(y_train)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
   # from imblearn.over_sampling import SMOTE
    # 假设 X 的形状是 (样本数, 时间步长, 特征数)
   # n_samples,  n_features, n_time_steps, =X_train.shape
   # X_reshaped = X_train.reshape(n_samples,  n_features *n_time_steps)
    # 使用 SMOTE 进行上采样
   # sm = SMOTE(random_state=42)
   # X_resampled, y_train = sm.fit_resample(X_reshaped, y_train)
    # 将上采样后的数据重塑回原来的三维格式
    #X_train= X_resampled.reshape(-1, n_features, n_time_steps)
    


    model = create_lstm_model(n1,n2)
    f1_callback = F1ScoreCallback(train_data=(X_train, y_train), val_data=(X_val, y_val))
    model.fit(X_train, y_train, epochs=20, batch_size=32,callbacks=[f1_callback])#callbacks=[f1_callback]
    
    
    sectionNames, strings, labels = [], [], []
    with jsonlines.open('C:/Users/liwei/Desktop/CS4248/test.jsonl') as f:
        for line in f.iter():
            sectionNames.append(line['sectionName'])
            strings.append(re.sub(r'\n', '. ', line['string']))
            labels.append(line['label'])
    strings = process_strings(strings)
    sectionNames = process_strings(sectionNames)
    
   # x_test = vectorise(strings, word2vec_model,n1)
    x_test = np.concatenate([vectorise(strings, word2vec_model,n1), vectorise(sectionNames, word2vec_model,n2)], axis=1)
    y_test = parse_label2index(labels)
    y_test = np.array(y_test)
    

    y_test_pred = np.argmax(model.predict(x_test), axis=-1)
    macro_f1_score = f1_score(y_test, y_test_pred, average='macro')
    print(macro_f1_score)
    report = classification_report(y_test, y_test_pred)
    print(report)

if __name__ == "__main__":
    main()