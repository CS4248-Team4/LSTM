from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from keras.layers import LSTM
from gensim.models import Word2Vec

import numpy as np
import re
import jsonlines

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from sklearn.metrics import f1_score

def vectorise(texts, w2v_model, max_length=40):
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
    strings_clean, num_citations = [], []
    for case in strings:
        case = re.sub(r'\[[0-9, ]*\]', '', case)
        case = re.sub(r'^...', '... ', case)
        open = False
        n = 0
        for c in case:
            if c == '(':
                open = True
                n += 1
            elif c == ')':
                open = False
            if (c == ';') and (open == True):
                n += 1
        case = word_tokenize(case.lower())
        strings_clean.append(case)
        num_citations.append(n)
    return strings_clean, num_citations

sec_name_mapping = {"discussion": 0, "introduction": 1, "unspecified": 2, "method": 3,
                    "results": 4, "experiment": 5, "background": 6, "implementation": 7,
                    "related work": 8, "analysis": 9, "conclusion": 10, "evaluation": 11,
                    "appendix": 12, "limitation": 13}

def process_sectionNames(sectionNames):
    returned = []
    for sectionName in sectionNames:
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
        returned.append(sec_name_mapping[newSectionName])
    return returned

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

def create_lstm_model():
    model = Sequential()
    #model.add(LSTM(128, input_shape=(33, 100), dropout=0.3, recurrent_dropout=0.3))
    model.add(Bidirectional(LSTM(64,dropout=0.3, recurrent_dropout=0.3), input_shape=(40, 100)))
    # 其他层...
    #model.add(Dense(32, activation='relu'))
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
    sectionNames, strings, labels, = [], [], []
    with jsonlines.open('scicite/train.jsonl') as f:
        for line in f.iter():
            sectionNames.append(line['sectionName'])
            strings.append(re.sub(r'\n', '. ', line['string']))
            labels.append(line['label'])
    strings, num_citations = process_strings(strings)

    word2vec_model = Word2Vec(sentences=strings, vector_size=100, window=5, min_count=1)
    word2vec_model.save('word2vec_model.bin')
    word2vec_model = Word2Vec.load('word2vec_model.bin')
   
    x_train = vectorise(strings, word2vec_model)
    sectionNames = process_sectionNames(sectionNames)
    y_train = parse_label2index(labels)
    y_train = np.array(y_train)
    
    sectionNames, strings, labels = [], [], []
    with jsonlines.open('scicite/dev.jsonl') as f:
        for line in f.iter():
            sectionNames.append(line['sectionName'])
            strings.append(re.sub(r'\n', '. ', line['string']))
            labels.append(line['label'])
    strings, num_citations = process_strings(strings)

    x_val = vectorise(strings, word2vec_model)
    sectionNames = process_sectionNames(sectionNames)
    y_val = parse_label2index(labels)
    y_val = np.array(y_val)

    model = create_lstm_model()
    f1_callback = F1ScoreCallback(train_data=(x_train, y_train), val_data=(x_val, y_val))
    model.fit(x_train, y_train, epochs=20, batch_size=32,callbacks=[f1_callback])

    sectionNames, strings, labels = [], [], []
    with jsonlines.open('scicite/test.jsonl') as f:
        for line in f.iter():
            sectionNames.append(line['sectionName'])
            strings.append(re.sub(r'\n', '. ', line['string']))
            labels.append(line['label'])
    strings, num_citations = process_strings(strings)

    x_test = vectorise(strings, word2vec_model)
    sectionNames = process_sectionNames(sectionNames)
    y_test = parse_label2index(labels)
    y_test = np.array(y_test)

    y_test_pred = np.argmax(model.predict(x_test), axis=-1)
    macro_f1_score = f1_score(y_test, y_test_pred, average='macro')
    print(macro_f1_score)
    report = classification_report(y_test, y_test_pred)
    print(report)

if __name__ == "__main__":
    main()