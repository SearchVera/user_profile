# numpy

# pandas

## 读数据

## 变换
1. 取出几列
```
X = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
```

2. 一列变换（性别变成0，1）
```
# Sex 
X.loc[X.Sex=='male','Sex'] = 1
X.loc[X.Sex == 'female','Sex'] = 0
# 该方法Sex为object类型
```

```
X['Sex'] = X.Sex.apply(lambda x: 1 if x == 'male' else 0)
```

3. 填充缺失值
```
X.Age = X.Age.fillna(int(X.Age.mean()))
#需要inplace
```

## 描述统计
1. 一列为空的有多少
```
train['Pclass'].isnull().sum()
```

2. 每一列的数据类型
```
X.dtypes
```
