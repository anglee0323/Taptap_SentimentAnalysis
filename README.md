# 2022年手游满意度调研
这是YNU2023春季课程《机器学习》的期末大作业。

本项目基于国内最大的移动手机游戏论坛TapTap，爬取手游用户评论，分析2022年手游市场的满意度。

## 项目结构

```
Sentiment Analysis

├── LICENSE //开源许可
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
│   └── transfer Learning //迁移学习
└── reoprt //报告
    └── report.tex //tex文件
    └── reoprt.pdf //pdf文件
```

## 数据来源

本次实验数据集来自[PP飞桨公开数据集](https://aistudio.baidu.com/aistudio/datasetdetail/183272)。

数据集包含手游网站 TapTap 上约 300 款游戏的标签评论，共4888个数据示例（按8：2的比例划分，3422条训练数据，1466条测试数据）

该数据集以csv格式存储，每一行含有`review`和`sentiment`两个参数，其中`review`为用户的评论文本，`sentiment`的值为 1 和 0 
> 用户评论低于 3 星（最多 5 星）被视为 0（不满意），其他为 1（满意）。两个类别的比例大致为1:1

```
【数据集结构】
taptap_review
├── test.csv #1466条数据
└── train.csv #3422条数据

【数据注释】
review：评论文本
sentiment：0代表不满意，1代表满意
```


## 算法部分

1.基于情感词典的情感极性分析

2.基于机器学习的情感极性分析
     
      (1)Bayes #朴素贝叶斯
     
      (2)svm #支持向量机
      
      (3)Adaboost #集成学习
3.基于深度学习的情感极性分析
      
      (1)LSTM模型
      
      (2)LSTM模型+Attention机制
3.基于迁移学习的情感极性分析
      
      (1)bert模型微调
      
      (2)Chatgpt模型微调
