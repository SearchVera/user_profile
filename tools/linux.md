
**md5**
```
md5sum file > file.md5
```

**cut**
```
cut -d ':' -f 1
```
-d ':' : 按冒号分割


**grep**
```
grep -5 -n "http" *.py
```
-5：显示前后5行
-n：显示行号
*.py：匹配所有.py文件

```
grep -- "->"
```
匹配'->'符号

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
