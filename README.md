# user_profile

## 论文
### NLP
- **Semi-supervised Sequence Learning**
> 摘要：使用自编码器和语言模型2种方法，使用无标记的数据来提升模型性能。训练完之后使用word embedding参数和LSTM权重来初始化监督模型的LSTM

> [代码](https://github.com/dongjun-Lee/transfer-learning-text-tf)

- **Recurrent neural network based language model**
> 摘要：最早提出基于rnn的语言模型：通过预测下一个词来构造损失函数

- **Transformer**
> [详解](https://medium.com/%E7%A8%8B%E5%BC%8F%E5%B7%A5%E4%BD%9C%E7%B4%A1/autoencoder-%E4%B8%89-self-attention-transformer-c37f719d222)

### Recommend
- **Deep Neural Network for YouTube Recommendation**
> 摘要：2016年youtube推荐系统，提出将推荐过程分为matching和ranking2个阶段

- **Wide & Deep Learning for Recommender Systems**
> 摘要：融合了LR模型和深度模型的优点
>> wide: 通过交叉特征能够有效记忆，但是需要人工工程，无法泛化到没出现的特征

>> deep: 通过embedding可以泛化到未出现过的特征，用于学习历史数据中不存在的特征组合

## 知识架构
### NLP
- [word2vec](https://zhuanlan.zhihu.com/p/26306795)
- [doc2vec]

### Graph
- **deepwalk**  [原理](https://zhuanlan.zhihu.com/p/56380812)

- **node2vec**

- SDNE

- [阿里应用](https://www.jianshu.com/p/229b686535f1)

### 无监督


## 工程
### tensorflow
- keras
    - Model
    - layers
        - Dense
        - LSTM
        - Input
        - Concatenate
        - Flatten
    - optimizer
        - Adam
    - metrics
        - Mean
        - CategoricalAccuracy
    - preprocessing
        - sequence
            - pad_sequences
    
        
        
- losses
    - CategoricalCrossentropy

- GradientTape

- train
    - Feature
    - Features
    - Example

- data
    - Dataset
    - experimental.TFRecordWriter
    - TFRecordDataset

- io
    - TFRecordWriter

- op
    - ones
    - zeros
    - concat
    - cast

