# -*- coding: utf-8 -*- 
# @Time : 6/26/23 18:49
# @Author : ANG

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate, Dot, Activation, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Multiply


# 读取CSV文件并加载数据
data = pd.read_csv('/Users/wallanceleon/Desktop/Sentiment Analysis/dataset/taptap_review.csv')

# 分离特征和标签
X = data.iloc[:, 0]  # 文本数据列
y = data.iloc[:, 1]  # 标签列

# 将标签编码为整数
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 文本数据处理
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

max_length = 100
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')

# 构建模型
embedding_dim = 100
hidden_units = 64

input_sequence = Input(shape=(max_length,))
embedding = Embedding(vocab_size, embedding_dim)(input_sequence)
lstm = LSTM(hidden_units, return_sequences=True)(embedding)

attention = Dense(1, activation='tanh')(lstm)
attention = Activation('softmax')(attention)
attention = Lambda(lambda x: K.mean(x, axis=1), name='attention_vec')(attention)
attention = RepeatVector(hidden_units)(attention)
attention = Permute([2, 1])(attention)

sent_representation = Multiply()([lstm, attention])
sent_representation = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)

output = Dense(1, activation='sigmoid')(sent_representation)

model = Model(inputs=input_sequence, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 模型训练
epochs = 50
batch_size = 64

model.fit(X_train_padded, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_padded, y_test))

# 模型评估
_, train_accuracy = model.evaluate(X_train_padded, y_train)
_, test_accuracy = model.evaluate(X_test_padded, y_test)

print('训练集正确率:', train_accuracy)
print('测试集正确率:', test_accuracy)
