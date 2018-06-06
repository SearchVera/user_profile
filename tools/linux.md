
**md5**
```
md5sum file > file.md5
```

**cut**
```
cut -d ':' -f 1
# -d ':' : 按冒号分割
```

**grep**
```
grep -5 -n "http" *.py
# -5：显示前后5行
# -n：显示行号
# *.py：匹配所有.py文件
```

```
grep -- "->"
# 匹配'->'符号
```

```
grep  "http" */*.py
# 在二级目录里查找
```


**su**
```
su work
# 切换到work用户，但不切换环境变量

su - work
# 切换到work用户，同时切换环境变量（建议使用）
```

**sort**
```
sort -r -n -k 2 -t : number.txt
# -r：倒序
# -n：按数字排序
# -k 2：按第二列排序
# -t : ：按':'分割
```

```
sort -k 1 -k 2 number.txt
# 先按第一列排序，再按第二列排序
```

**du：磁盘占用**
```
du -sh *
```


**awk**
```
awk '{sum += $1};END {print sum}' file
```

**sed**
```
sed -n "3,6p" file
# 显示3-6行
```

```
sed "s/lds/ld/g"
# 全局将lds替换为ld
```

**chmod**
```
chmod a+x log2012.log
# a:权限范围(a=所有用户及群组;u=当前用户;g=当前组;o=其他)
# x:权限(x=执行权限;r=读权限;w=写权限)
```
