# 2022年手游市场满意度调研

本项目基于国内最大的移动手机游戏论坛TapTap，爬取手游用户评论，使用多种方法分析2022年手游市场的满意度。


## 目录
- [一、快速开始](#一快速开始)
- [二、项目结构](#二项目结构)
- [三、数据集说明](#三数据集说明)
- [四、模型选择](#四模型选择)
- [五、实验结果](#五实验结果)
- [六、版权声明](#六版权声明)


## 一、快速开始
1. 克隆项目到本地
```
git clone https://github.com/anglee2002/SentimentAnalysis.git
```
2. 配置python环境并安装相关依赖(建议使用conda环境)
```
# 在项目根目录下执行:
# 创建conda环境,并指定python版本为3.8及以上
conda create -n sentimentanalysis python=3.8
# 激活环境
conda activate sentimentanalysis
# 安装依赖
pip install -r requirements.txt
```
3. 运行程序
```
请更改src目录下的python程序中的数据集路径,然后在项目根目录下执行:

python src/*/*.py
```
4. 附加说明
```
1. 本实验设备为MacBook Pro 2020(m1 Apple silicon)，系统为macOS Ventura 13.2 (22D49)。
   由于硬件兼容问题,没有测试Windows系统,但理论上可以运行。
2. 本实验的tensorflow为ARM版本,由于tensorflow的原因,可能会出现一些警告,但是不影响运行。
3. 本实验的bert模型为中文版,由于模型文件过大,没有上传,请自行下载。
4. 若运行出现问题,请在issue中提出,或联系473394227@mail.ynu.edu.cn。
```


## 二、项目结构
```
Sentiment Analysis

├── LICENSE //开源许可
├── README.md //README文档
├── requirements.txt //依赖说明
├── dataset //数据集
│   └── taptap_review //TapTap评论数据集
├── src //源代码
│   └── sentiment_dictionary //情感字典
│       └── sentiment_dictionary.py //情感字典python程序
│       └── 情感极性词典
│           └── 中文停用词.txt
│           └── 否定词.txt
│           └── 程度副词.txt
│           └── BosonNLP //boson情感词典
│               └── BosonNLP_sentiment_score.txt
│               └── license.txt 
│               └── README.txt
│           └── 词典来源说明.txt
│   └── machine_learning //机器学习
│           └── svm.py //支持向量机
│           └── naive_bayes.py //朴素贝叶斯
│           └── adaboost.py //集成学习
│   └── deep_learning //深度学习
│       └── bi_lstm.py //Bi-LSTM模型
│       └── attention_lstm.py //LSTM模型+Attention机制
│   └── transfer Learning //迁移学习
│       └── finetune_bert.py //bert模型微调
├── reoprt //报告
│   └── 开题报告（由于数据集问题，以弃用该方案，请参考终期报告）
│   └── 终期报告
│       └── fig //报告中的图片
│       └── reference.bib //参考文献
│       └── report.tex //LaTeX源文件
│       └── 基于多种方式实现的2022年中国手机游戏满意度分析.pdf
├── 参考文献 
└── 答辩ppt
            
```


## 三、数据集说明

本次实验数据集来自[PP飞桨公开数据集](https://aistudio.baidu.com/aistudio/datasetdetail/183272)。

数据集包含手游网站 TapTap 上约 300 款游戏的标签评论，共4888个数据示例。

该数据集以csv格式存储，每一行含有`review`和`sentiment`两个参数，其中`review`为用户的评论文本，`sentiment`的值为 1 和 0 
> 用户评论低于 3星(最多5星)被视为 0(不满意)，其他为 1(满意)。两个类别的比例大致为 1:1

```
【数据集结构】
taptap_review # 4888条数据,按照7:3划分训练集和测试集
├── train.csv #3422条数据
└── test.csv #1466条数据

【数据注释】
review：评论文本
sentiment：0代表不满意，1代表满意
```


## 四、模型选择

1.基于情感词典的情感极性分析

      (1)基于BosonNLP的情感分析
2.基于机器学习的情感极性分析
     
      (1)Svm #支持向量机
     
      (2)Naive_bayes #朴素贝叶斯
      
      (3)Adaboost #集成学习
3.基于深度学习的情感极性分析
      
      (1)Bi-LSTM模型
      
      (2)LSTM模型+Attention机制
4.基于迁移学习的情感极性分析
      
      (1)bert模型微调

## 五、实验结果

![](https://raw.githubusercontent.com/anglee2002/Picbed/main/screenshot2023-06-30%2003.32.35.png)
## 六、版权声明

BosonNLP情感词典由[玻森数据](https://bosonnlp.com/)授权使用。

Bert模型版权由[谷歌公司](https://www.google.com/)所有。

其他部分根据[MIT License](LICENSE)开源。
