## spark基本语法

### 创建
1. 从文件中创建
```
val lines = sc.textFile("/usr/local/Cellar/apache-spark/2.3.1/README.md")
```

2. parallelize
```
scala> val l = List("spark","hadoop","spark")
l: List[String] = List(spark, hadoop, spark)

scala> val rdd = sc.parallelize(l)
rdd: org.apache.spark.rdd.RDD[String] = ParallelCollectionRDD[5] at parallelize at <console>:26
```
