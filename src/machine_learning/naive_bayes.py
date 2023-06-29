# -*- coding: utf-8 -*- 
# @Time : 6/19/23 13:29 
# @Author : ANG

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 读取CSV文件并加载数据
data = pd.read_csv('/Users/wallanceleon/Desktop/Sentiment Analysis/dataset/taptap_review.csv')

# 拆分数据集为训练集和测试集
X = data.iloc[:, 0]  # 文本数据列
y = data.iloc[:, 1]  # 标签列

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征提取
vectorizer = TfidfVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# 训练朴素贝叶斯分类器
nb_model = MultinomialNB()
nb_model.fit(X_train_features, y_train)

# 在训练集上进行预测并计算正确率
train_predictions = nb_model.predict(X_train_features)
train_accuracy = accuracy_score(y_train, train_predictions)
print("训练集正确率：", train_accuracy)

# 在测试集上进行预测并计算正确率
test_predictions = nb_model.predict(X_test_features)
test_accuracy = accuracy_score(y_test, test_predictions)
print("测试集正确率：", test_accuracy)
