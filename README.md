
## 知识架构
### 基础模型
- **Logistic Regression**
- **决策树**
- **SVM**
- **集成：Boosting, Bagging**
- **GBDT**
- **xgboost**
- **随机森林**

### 数学基础
- **特征值，特征向量**
- **奇异值分解, PCA**
- **pearson系数**
- **协方差矩阵**
= **alias method采样算法** [解决均匀分布生成其他分布的方法](https://blog.csdn.net/qq_33765907/article/details/79182355)

### 工程基础
- **偏差（bias）,方差（var）**
- **precision, recall, accuracy**
- **ROC, AUC**
- **正则项**
- **Dropout**
- **early stop**
- **梯度消失** relu, batch norm, residual net, w initialize
- **梯度更新算法** Momentom（指数加权平均）, RMSProp, Adam
- **decay learning rate**
- **batch norm**
- **残差网络**
- **调参顺序** learning rate -> 正则系数 -> batch size -> hidden size 
- **特征工程**
    - 缺失值：忽略，中心值填充
    - 噪声：分箱光滑
    - 异常值：去除
    - 归一化，标准化，正则化：[sklearn.preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
    - 离散化, 数据变换(log)
    - 特征相关性分析

### 深度模型
- **MLP**
- **LSTM** [详解](https://www.jianshu.com/p/95d5c461924c)
- **GRU**
- **卷积，pooling**


### NLP
- **word2vec** [原理](https://zhuanlan.zhihu.com/p/26306795)
- **doc2vec**
- **LDA**
- **ELMO**
- **GPT**
- **Bert** [综述](https://zhuanlan.zhihu.com/p/49271699)
- **Transformer** [详解](https://medium.com/%E7%A8%8B%E5%BC%8F%E5%B7%A5%E4%BD%9C%E7%B4%A1/autoencoder-%E4%B8%89-self-attention-transformer-c37f719d222)

### Graph
- **deepwalk**  [原理](https://zhuanlan.zhihu.com/p/56380812)
- **node2vec**
- **SDNE** [原理](https://zhuanlan.zhihu.com/p/56637181)

### 无监督
- **DBSCAN**
- **TSNE**

### RecSys
- **基于内容**
- **协同过滤**
- **深度模型**
- **排序**

## 论文
### Recommend
- **Deep Neural Network for YouTube Recommendation**
    - 摘要：2016年youtube推荐系统，提出将推荐过程分为matching和ranking2个阶段

- **Wide & Deep Learning for Recommender Systems**
    - 摘要：融合了LR模型和深度模型的优点
    - wide: 通过交叉特征能够有效记忆，但是需要人工工程，无法泛化到没出现的特征
    - deep: 通过embedding可以泛化到未出现过的特征，用于学习历史数据中不存在的特征组合

- **Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate**
    - 摘要：同时优化CTR和CVR
    - [精读](https://blog.csdn.net/sinat_15443203/article/details/83713802）

- **Real-time Personalization using Embeddings for Search Ranking at Airbnb**
    - 摘要：对用户的浏览行为构建embedding，综合考虑了最终的预定结果和地区
    - [精读](https://blog.csdn.net/like_red/article/details/88389918)

### NLP
- **Semi-supervised Sequence Learning**
    - 摘要：使用自编码器和语言模型2种方法，使用无标记的数据来提升模型性能。训练完之后使用word embedding参数和LSTM权重来初始化监督模型的LSTM
    - [代码](https://github.com/dongjun-Lee/transfer-learning-text-tf)

- **Recurrent neural network based language model**
    - 摘要：最早提出基于rnn的语言模型：通过预测下一个词来构造损失函数
