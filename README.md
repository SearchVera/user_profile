# user_profile

## 论文
### NLP
1. **Semi-supervised Sequence Learning**
> 摘要：使用自编码器和语言模型2种方法，使用无标记的数据来提升模型性能。训练完之后使用word embedding参数和LSTM权重来初始化监督模型的LSTM

> [代码](https://github.com/dongjun-Lee/transfer-learning-text-tf)

2. **Recurrent neural network based language model**
> 摘要：最早提出基于rnn的语言模型：通过预测下一个词来构造损失函数

### Recommend
1. **Deep Neural Network for YouTube Recommendation**
> 摘要：2016年youtube推荐系统，提出将推荐过程分为matching和ranking2个阶段

## 工程
### tensorflow
- keras
    - Model
    - layers
        - Dense
        - LSTM
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
    - Example

- data
    - Dataset
    - experimental.TFRecordWriter
    - TFRecordDataset

- op
    - ones
    - zeros
    - concat
    - cast

