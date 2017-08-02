# K临近算法（KNN）
## k临近算法概述
k临近算法采用裁量不同特征值之间的距离方法进行分类
> 优点

	精度高、对异常值不敏感、无数据输入假定

> 缺点

	计算复杂度高、空间复杂度高

## 代码中函数功能记录
#### 1. numpy.tile()函数

		In [15]: a = [1, 2]
		In [16]: np.tile(a,2)
		Out[16]: array([1, 2, 1, 2])
		In [19]: np.tile(a,(2,1))
		Out[19]:array([[1, 2],
					[1, 2]])
		In [20]: np.tile(a,(1,2))
		Out[20]: array([[1, 2, 1, 2]])
#### 2. operator.itemgetter函数
operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号（即需要获取的数据在对象中的序号），下面看例子。
	
		a = [1,2,3] 
		>>> b=operator.itemgetter(1)  //定义函数b，获取对象的第2个域的值（从0开始）
		>>> b(a) 
		2 
		>>> a = [[1,2,3],[4,5,6]]
		>>> b(a)
		[4,5,6]
		>>> b=operator.itemgetter(1,0)
		>>> b(a)
		[4,5,6],[1,2,3]

要注意，operator.itemgetter函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值

#### 3. sorted函数
Python内置的排序函数sorted可以对list或者iterator进行排序
该函数原型为：

sorted(iterable[, cmp[, key[, reverse]]])

参数解释：

（1）iterable指定要排序的list或者iterable，不用多说；

（2）cmp为函数，指定排序时进行比较的函数，可以指定一个函数或者lambda函数，如：students为类对象的list，没个成员有三个域，用sorted进行比较时可以自己定cmp函数，

例如这里要通过比较第三个数据成员来排序，代码可以这样写：

		students = [('john', 'A', 15), ('jane', 'B',12), ('dave', 'B', 10)]
		sorted(students, key=lambda student : studen[2])
（3）key为函数，指定取待排序元素的哪一项进行排序，函数用上面的例子来说明，代码如下：

      sorted(students, key=lambda student : student[2])
      
key指定的lambda函数功能是去元素student的第三个域（即student[2]），因此sorted排序时，会以students所有元素的第三个域来进行排序。

有了上面的operator.itemgetter函数，也可以用该函数来实现，

例如要通过student的第三个域排序，可以这么写：

	sorted(students, key=operator.itemgetter(2)) 

sorted函数也可以进行多级排序，例如要根据第二个域和第三个域进行排序，可以这么写：

	sorted(students, key=operator.itemgetter(1,2))

即先跟句第二个域排序，再根据第三个域排序。

（4）reverse参数就不用多说了，是一个bool变量，表示升序还是降序排列，默认为false（升序排列），定义为True时将按降序排列。
#### 4. numpy.shape
得到矩阵或数组的纬度
#### 5. matplotlib官方文档地址
[http://matplotlib.org/2.0.2/api/_as_gen/](matplotlib)