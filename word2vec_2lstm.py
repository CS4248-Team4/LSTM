# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 04:33:09 2024

@author: liwei
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam


def vectorise(texts, w2v_model, max_length):
    vector_size = w2v_model.vector_size
    texts_vec = []

    words = set(w2v_model.wv.index_to_key)

    for text in texts:
        sentence_vectors = []

        for word in text[:max_length]:
            if word in words:
                sentence_vectors.append(w2v_model.wv[word])
            else:
                sentence_vectors.append(np.zeros(vector_size))

        for _ in range(max_length - len(sentence_vectors)):
            sentence_vectors.append(np.zeros(vector_size))
        texts_vec.append(sentence_vectors)

    return np.array(texts_vec)


def process_strings(strings):
    returned = []
    for case in strings:
        if not isinstance(case, str):  
            case = str(case)  
        case = re.sub(r'\[[0-9, ]*\]', '', case)
        case = re.sub(r'^...', '... ', case)
        case = word_tokenize(case.lower())
        if not case:
            case = [' ']
        returned.append(case)
    return returned

def preprocess_sectionName(sectionName):
    sectionName = str(sectionName)
    newSectionName = sectionName.lower()

    if newSectionName != None:
        if "introduction" in newSectionName or "preliminaries" in newSectionName:
            newSectionName = "introduction"
        elif "result" in newSectionName or "finding" in newSectionName:
            newSectionName = "results"
        elif "method" in newSectionName or "approach" in newSectionName:
            newSectionName = "method"
        elif "discussion" in newSectionName:
            newSectionName = "discussion"
        elif "background" in newSectionName:
            newSectionName = "background"
        elif "experiment" in newSectionName or "setup" in newSectionName or "set-up" in newSectionName or "set up" in newSectionName:
            newSectionName = "experiment"
        elif "related work" in newSectionName or "relatedwork" in newSectionName or "prior work" in newSectionName or "literature review" in newSectionName:
            newSectionName = "related work"
        elif "evaluation" in newSectionName:
            newSectionName = "evaluation"
        elif "implementation" in newSectionName:
            newSectionName = "implementation"
        elif "conclusion" in newSectionName:
            newSectionName = "conclusion"
        elif "limitation" in newSectionName:
            newSectionName = "limitation"
        elif "appendix" in newSectionName:
            newSectionName = "appendix"
        elif "future work" in newSectionName or "extension" in newSectionName:
            newSectionName = "appendix"
        elif "analysis" in newSectionName:
            newSectionName = "analysis"
        else:
            newSectionName = "unspecified"
        return newSectionName

def parse_label2index(labels):
    index = []
    for i in range(len(labels)):
        label = labels.iloc[i] 
        if label == "background":
            index.append(0)
        elif label == "method":
            index.append(1)
        else: 
            index.append(2)
    return index



def create_lstm_model(n1, n2, vector_size):
    text_input = Input(shape=(n1, vector_size), name='text_input')
    lstm_out = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(text_input)
    #LSTM(64, dropout=0.2, recurrent_dropout=0.2)(text_input)
    #

    section_input = Input(shape=(n2,), name='section_input')  # n2 after get_dummies

    concatenated = Concatenate()([lstm_out, section_input])

    dense1 = Dense(16, activation='relu')(concatenated)
    output = Dense(3, activation='softmax')(dense1)  # 3 class

    model = Model(inputs=[text_input, section_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

class F1ScoreCallback(Callback):
    def __init__(self, train_data, val_data):
        super(F1ScoreCallback, self).__init__()  # 确保正确调用超类的构造函数
        self.train_data = train_data
        self.val_data = val_data
        self.train_f1_scores = []  # 初始化用于存储训练集F1分数的列表
        self.val_f1_scores = []  # 初始化用于存储验证集F1分数的列表

    def on_epoch_end(self, epoch, logs=None):
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        y_train_pred = np.argmax(self.model.predict([X_train[0], X_train[1]]), axis=-1)
        y_val_pred = np.argmax(self.model.predict([X_val[0], X_val[1]]), axis=-1)

        
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        self.train_f1_scores.append(train_f1)
        self.val_f1_scores.append(val_f1)
        
        print(f'Epoch {epoch+1} - train F1: {train_f1:.4f}, val F1: {val_f1:.4f}')

    def on_train_end(self, logs=None):
        
        plt.plot(self.train_f1_scores, label='Train F1')
        plt.plot(self.val_f1_scores, label='Validation F1')
        plt.title('F1 Score Trend')
        plt.ylabel('F1 Score')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

def main():
    word2vec_model = Word2Vec.load('C:/Users/liwei/Desktop/CS4248/word2vec_model.bin')
        
    import json
    import pandas as pd
    import json

    with open('C:/Users/liwei/Desktop/CS4248/train.jsonl', 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    Df_train = pd.DataFrame(train_data)


    with open('C:/Users/liwei/Desktop/CS4248/test.jsonl', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    Df_test = pd.DataFrame(test_data)
    
    df_train = Df_train[['string', 'sectionName', 'label','label_confidence','isKeyCitation']]
    df_test = Df_test[['string', 'sectionName', 'label','label_confidence','isKeyCitation']]
    
    #df_train = df_train.dropna(subset=['label_confidence', 'isKeyCitation'])#
    #df_train.dropna(subset = 'isKeyCitation')
    
    all_categories = [
    "introduction", "results", "method", "discussion", "background", 
    "experiment", "related work", "evaluation", "implementation", 
    "conclusion", "limitation", "appendix", "analysis", "unspecified"
    ]
    
    X_train_strings = process_strings(df_train['string'])
    n1 = 33
    X_train = vectorise(X_train_strings, word2vec_model,n1)
    
    
    X_train_sectionName = df_train['sectionName'].apply(preprocess_sectionName)
    X_train_sectionName = pd.get_dummies(X_train_sectionName)
    X_train_sectionName = X_train_sectionName.reindex(columns=all_categories, fill_value=0)
    X_train_sectionName = X_train_sectionName.astype(int)
    X_train_sectionName = np.array(X_train_sectionName)
    
    X_train_label_confidence = np.array(df_train['label_confidence'])
    
    X_train_isKeyCitation = pd.get_dummies(df_train['isKeyCitation'])
    X_train_isKeyCitation = X_train_isKeyCitation.astype(int)
    X_train_isKeyCitation = np.array(X_train_isKeyCitation)
    X_train_isKeyCitation =  X_train_isKeyCitation[:,0]
    
    X_train_isKeyCitation = X_train_isKeyCitation.reshape(-1, 1)
    X_train_label_confidence = X_train_label_confidence.reshape(-1, 1)
    
    X_train_other =X_train_sectionName
    #X_train_other =np.hstack((X_train_sectionName,X_train_isKeyCitation))#,  X_train_isKeyCitation,X_train_label_confidence
    #,X_train_label_confidence
    y_train = parse_label2index(df_train['label'])
    y_train = np.array(y_train)
    
    X_train, X_val,X_train_other,X_val_other, y_train, y_val = train_test_split(X_train, X_train_other,y_train, test_size=0.1, random_state=42)    

    model = create_lstm_model(n1, X_train_other.shape[1], 100) 
    f1_callback = F1ScoreCallback(train_data=([X_train, X_train_other],y_train), val_data=([X_val, X_val_other],y_val))
    model.fit([X_train, X_train_other], y_train, epochs=20, batch_size=32, callbacks=[f1_callback])
    
    
    X_test_strings = process_strings(df_test['string'])
    n1 = 33
    X_test = vectorise(X_test_strings, word2vec_model,n1)
    
    
    X_test_sectionName = df_test['sectionName'].apply(preprocess_sectionName)
    X_test_sectionName = pd.get_dummies(X_test_sectionName)
    X_test_sectionName = X_test_sectionName.reindex(columns=all_categories, fill_value=0)
    X_test_sectionName = X_test_sectionName.astype(int)
    X_test_sectionName = np.array(X_test_sectionName)
    
    X_test_label_confidence = np.array(df_test['label_confidence'])
    
    X_test_isKeyCitation = pd.get_dummies(df_test['isKeyCitation'])
    X_test_isKeyCitation = X_test_isKeyCitation.astype(int)
    X_test_isKeyCitation = np.array(X_test_isKeyCitation)
    X_test_isKeyCitation =  X_test_isKeyCitation[:,0]
    
    X_test_isKeyCitation = X_test_isKeyCitation.reshape(-1, 1)
    X_test_label_confidence = X_test_label_confidence.reshape(-1, 1)
    #
    X_test_other = X_test_sectionName
    #X_test_other =np.hstack((X_test_sectionName,X_test_isKeyCitation)) #X_test_label_confidence,, X_test_isKeyCitation,X_test_label_confidence
    #,X_test_label_confidence
    
    y_test = parse_label2index(df_test['label'])
    y_test = np.array(y_test)
    
    
    y_pred_encoded = model.predict([X_test, X_test_other])
    y_test_pred = np.argmax(y_pred_encoded, axis=1)
    macro_f1_score = f1_score(y_test, y_test_pred, average='macro')
    print(macro_f1_score)
    report = classification_report(y_test, y_test_pred)
    print(report)

if __name__ == "__main__":
    main()