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

### 常用操作
1. reduceByKey(func)
```
scala> val rdd_c = rdd.map(word => (word,1))
rdd_c: org.apache.spark.rdd.RDD[(String, Int)] = MapPartitionsRDD[6] at map at <console>:25

scala> rdd_c.reduceByKey((a,b) => a+b).foreach(println)
(spark,2)
(hadoop,1)
```

2. groupByKey()
```
scala> rdd_c.groupByKey().foreach(println)
(spark,CompactBuffer(1, 1))
(hadoop,CompactBuffer(1))
```

3. sortByKey()
```
scala> rdd_c.sortByKey().foreach(println)
(spark,1)
(spark,1)
(hadoop,1)
```

4. mapValues()
```
scala> rdd_c.mapValues(x => x+1).foreach(println)
(spark,2)
(hadoop,2)
(spark,2)
```

5. join：算笛卡尔乘积
```
scala> rdd_c.foreach(println)
(spark,1)
(hadoop,1)
(spark,1)

scala> rdd_d.foreach(println)
(spark,4)
(hadoop,2)

scala> rdd_c.join(rdd_d).foreach(println)
(spark,(1,4))
(spark,(1,4))
(hadoop,(1,2))
```
