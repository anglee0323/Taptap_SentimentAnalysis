# YNUer情感极性分析
这是YNU2023春季课程《机器学习》的期末大作业。

本项目基于云南大学校园集市的发贴情况，分析云大师生的精神状态以及对于校园热点问题的情感倾向。

__项目结构:__

```
Sentiment Analysis

├── LICENSE //开源证书
├── README.md //README文档
├── requirements.txt //依赖说明
├── main.py //启动脚本
├── dataset //数据集
├── src //源代码
│   └── __init__.py //配置文件
│   └── data_processing //数据处理
│   └── sentiment_dictionary //情感字典
│   └── machine_learning //机器学习
│   └── deep_learning //深度学习
│   └── verification //验证
└── reoprt //报告
    └── report.tex //tex文件
    └── reoprt.pdf //pdf文件
```

__任务说明__:

__通过网页爬虫获取基本数据，通过清洗，标注等步骤，形成完备可用的数据集；使用机器学习和深度学习的方法训练模型，得到所需分类器；通过调用成熟的商业接口验证分类的准确性。具体说明如下：__

__数据来源__：爬取云南大学校园集市平台发帖信息

__数据处理算法__：

1.基于情感词典的情感极性分析

2.基于机器学习的情感极性分析
     
      (1)Bayes #朴素贝叶斯
     
      (2)svm #支持向量机
      
      (3)Adaboost #集成学习
3.基于深度学习的情感极性分析
      
      (1)LSTM模型
      
      (2)LSTM模型+Attention机制
      
__模型验证__：

      (1)基于OpenAI gpt3.5 turbo的模型验证
      
      (2)基于百度飞桨PaddleNLP的模型验证


