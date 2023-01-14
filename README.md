# DataMine

* Python 语言与数据挖掘工具
* 数据挖掘的基本方法
* 可视化
* 分类
* 聚类
* 关联规则（亲和性分析）
* 案例分析

数据分析用pandas和matplotlib
数据挖掘根据理论自己实现 apriori，kmeans，决策树 等数据挖掘算法
可以作为入门学习用

商品库存数据分析与挖掘
背景
分析与挖掘商品数据，充分了解数据
导入工具
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data_Mine import apriori,kmeans,decision_tree
导入数据
data=pd.read_csv('example.csv')
data
A	B	C	D	E	F	G	H	I	J	K	L	M	N	O
0	开发区分店	杯子	阿塞拜疆	2020	4	20	工具	5	99	39	57	高	高	高	是
1	天山区分店	咖啡	尼日尔	2020	4	15	食品	17	39	23	49	低	低	高	否
2	开发区分店	口罩	以色列	2019	5	5	其它	16	2	42	57	低	高	高	否
3	天山区分店	巧克力	马提尼克	2019	1	2	食品	25	84	45	56	高	高	高	是
4	米东区	消毒水	波多黎各	2020	10	23	其它	11	44	45	51	低	高	高	否
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
217	新市区分店	牛奶	圭亚那	2020	11	24	食品	5	33	32	48	低	低	低	否
218	米东区	扫把	加蓬	2019	5	7	工具	16	6	39	53	低	高	高	否
219	开发区分店	钳子	蒙古	2020	8	16	工具	15	65	45	52	高	高	高	是
220	水磨沟区分店	啤酒	瓦里斯和富士那群岛	2019	5	8	食品	16	19	26	47	低	低	低	否
221	开发区分店	啤酒	瓦里斯和富士那群岛	2020	4	1	食品	7	66	25	50	高	低	高	否
222 rows × 15 columns

数据预处理
data.drop_duplicates(inplace=True) 
# 数据信息
data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 222 entries, 0 to 221
Data columns (total 15 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   A       222 non-null    object
 1   B       222 non-null    object
 2   C       222 non-null    object
 3   D       222 non-null    int64 
 4   E       222 non-null    int64 
 5   F       222 non-null    int64 
 6   G       222 non-null    object
 7   H       222 non-null    int64 
 8   I       222 non-null    int64 
 9   J       222 non-null    int64 
 10  K       222 non-null    int64 
 11  L       222 non-null    object
 12  M       222 non-null    object
 13  N       222 non-null    object
 14  O       222 non-null    object
dtypes: int64(7), object(8)
memory usage: 27.8+ KB
data.describe()
D	E	F	H	I	J	K
count	222.000000	222.000000	222.000000	222.000000	222.000000	222.000000	222.000000
mean	2019.477477	6.288288	14.274775	15.995495	53.954955	35.121622	48.810811
std	0.500621	3.281340	8.461849	8.818685	29.647823	7.646522	4.868429
min	2019.000000	1.000000	1.000000	1.000000	1.000000	18.000000	40.000000
25%	2019.000000	3.000000	7.000000	8.000000	31.000000	28.000000	45.000000
50%	2019.000000	6.000000	14.000000	17.000000	57.000000	37.500000	49.000000
75%	2020.000000	9.000000	22.000000	24.000000	80.000000	41.000000	53.000000
max	2020.000000	12.000000	28.000000	30.000000	100.000000	48.000000	58.000000
数据可视化
单变量数据分布分析
data1=data.groupby('A',as_index=False).agg(数量=('A','count'))
data1
A	数量
0	天山区分店	44
1	开发区分店	39
2	新市区分店	30
3	水磨沟区分店	34
4	沙区分店	35
5	米东区	40
plt.figure(figsize=(15,8))
plt.pie(data1['数量'],labels=data1['A'],autopct='%.2f%%',shadow=True)
plt.title('分店数分布')
plt.show()

双变量的关联分析
data2=data.groupby('B',as_index=False).agg(均价=('H','mean'),最高价=('H','max'),最低价=('H','min'))
data2
