import pandas
from pandas import read_excel
import  numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



df= pd.read_csv('D:\MEFFGRN\EPCdata\gene_expression_ko04622-top.csv', header=0, index_col=0)
data = df.values  # 将dataframe转化为array
values = data.astype('float32')  # 定义数据类型


# 标准化
tool = MinMaxScaler(feature_range=(0, 1))
data = tool.fit_transform(data)



data=pd.DataFrame(data)
#df = pandas.DataFrame(data)  # 将array转化为dataframe


df.columns = data.columns # 命名标题行
data.to_csv('D:\MEFFGRN\EPCdata\gene_expression_ko04622-top1.csv')  # 另存为excel（删除索引）
