# -*- coding: utf-8 -*-
# @Time : 6/16/23 12:52
# @Author : ANG

import csv
from collections import defaultdict

import jieba

# jieba分词后去除停用词
def seg_word(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = []
    for i in seg_list:
        seg_result.append(i)
    stopwords = set()
    with open('/Users/wallanceleon/Desktop/Sentiment Analysis/src/sentiment_dictionary/情感极性词典/中文停用词.txt', 'r') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x: x not in stopwords, seg_result))

# 找出文本中的情感词、否定词和程度副词
def classify_words(word_list):
    # 读取情感词典文件
    sen_file = open('/Users/wallanceleon/Desktop/Sentiment Analysis/src/sentiment_dictionary/情感极性词典/BosonNLP/BosonNLP_sentiment_score.txt', 'r+', encoding='utf-8')
    # 获取词典文件内容
    sen_list = sen_file.readlines()
    # 创建情感字典
    sen_dict = defaultdict()
    # 读取词典每一行的内容，将其转换成字典对象，key为情感词，value为其对应的权重
    for i in sen_list:
        if len(i.split(' ')) == 2:
            sen_dict[i.split(' ')[0]] = i.split(' ')[1]

    # 读取否定词文件
    not_word_file = open('/Users/wallanceleon/Desktop/Sentiment Analysis/src/sentiment_dictionary/情感极性词典/否定词.txt', 'r+', encoding='utf-8')
    not_word_list = not_word_file.readlines()
    # 读取程度副词文件
    degree_file = open('/Users/wallanceleon/Desktop/Sentiment Analysis/src/sentiment_dictionary/情感极性词典/程度副词.txt', 'r+')
    degree_list = degree_file.readlines()
    degree_dict = defaultdict()
    for i in degree_list:
        degree_dict[i.split(',')[0]] = i.split(',')[1]

    sen_word = dict()
    not_word = dict()
    degree_word = dict()
    # 分类
    for i in range(len(word_list)):
        word = word_list[i]
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dict.keys():
            # 找出分词结果中在情感字典中的词
            sen_word[i] = sen_dict[word]
        elif word in not_word_list and word not in degree_dict.keys():
            # 分词结果中在否定词列表中的词
            not_word[i] = -1
        elif word in degree_dict.keys():
            # 分词结果中在程度副词中的词
            degree_word[i] = degree_dict[word]

    # 关闭打开的文件
    sen_file.close()
    not_word_file.close()
    degree_file.close()
    # 返回分类结果
    return sen_word, not_word, degree_word

# 计算情感词的分数
def score_sentiment(sen_word, not_word, degree_word, seg_result):
    # 权重初始化为1
    W = 1
    score = 0
    # 情感词下标初始化
    sentiment_index = -1
    # 情感词的位置下标集合
    sentiment_index_list = list(sen_word.keys())
    # 遍历分词结果
    for i in range(0, len(seg_result)):
        # 如果是情感词
        if i in sen_word.keys():
            # 权重*情感词得分
            score += W * float(sen_word[i])
            # 情感词下标加一，获取下一个情感词的位置
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list) - 1:
                # 判断当前的情感词与下一个情感词之间是否有程度副词或否定词
                for j in range(sentiment_index_list[sentiment_index], sentiment_index_list[sentiment_index + 1]):
                    # 更新权重，如果有否定词，权重取反
                    if j in not_word.keys():
                        W *= -1
                    elif j in degree_word.keys():
                        W *= float(degree_word[j])
        # 定位到下一个情感词
        if sentiment_index < len(sentiment_index_list) - 1:
            i = sentiment_index_list[sentiment_index + 1]
    return score

# 计算得分
def sentiment_score(sentence):
    # 1.对文档分词
    seg_list = seg_word(sentence)
    # 2.将分词结果转换成字典，找出情感词、否定词和程度副词
    sen_word, not_word, degree_word = classify_words(seg_list)
    # 3.计算得分
    score = score_sentiment(sen_word, not_word, degree_word, seg_list)
    return score

def process_csv(file_path):
    sentences = []
    labels = []

    with open(file_path, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # 跳过第一行

        for row in csv_reader:
            sentence = row[0]
            label = int(row[1])

            sentences.append(sentence)
            labels.append(label)

    # 使用sentiment_score(),计算每个句子的情感分数
    scores = []
    for sentence in sentences:
        score = sentiment_score(sentence)
        scores.append(score)

    return scores, labels

def calculate_accuracy(predictions, labels):
    correct_count = 0
    total_count = len(predictions)

    for pred, label in zip(predictions, labels):
        if pred > 0 and label == 1:  # Positive sentiment
            correct_count += 1
        elif pred <= 0 and label == 0:  # Negative sentiment
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

if __name__ == "__main__":
    # CSV文件路径
    csv_file_path = "/Users/wallanceleon/Desktop/Sentiment Analysis/dataset/taptap_review.csv"

    # 处理CSV文件
    print("Processing CSV file...")
    scores, labels = process_csv(csv_file_path)
    print("CSV file processed successfully.")

    # 计算正确率
    print("Calculating accuracy...")
    accuracy = calculate_accuracy(scores, labels)

    # 打印正确率
    print("Accuracy: {:.2%}".format(accuracy))
