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
