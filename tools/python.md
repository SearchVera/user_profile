## virtualenv使用
```
virtualenv --system-site-packages ~/tensorflow
# 为tensorflo文件夹创建一个环境

source bin/activate
# 激活

deactivate
# 关闭环境
```

## numpy
**(5,)变成(5,1)：newaxis**
```
b = np.linspace(-1,1,5)
##b.shape = (5,)

b=np.linspace(-1,1,5)[:, np.newaxis]
##b.shape = (5,1)
```
**linspace：均匀取值**
```
np.linspace(-1, 1, 100)
# [-1,1]之间均匀取100个点
```

**产生随机数**

```
noise = np.random.normal(0, 0.01, x.shape)
# 0=均值 0.01=方差 x.shape=大小
```
## 其他
**pip查看版本号**
```
pip list
```
