# -*- coding:utf-8-*-
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

data = pd.read_excel('data/mid_data.xlsx')
print(data)

inputer1 = IterativeImputer(estimator=RandomForestRegressor(n_estimators=300))
inputer2 = IterativeImputer(estimator=LinearRegression())
inputer3 = KNN(k=3)
inputer4 = SoftImpute()

# #随机森林回归填补缺失值
# x_RF = inputer1.fit_transform(data)
# data_RF = pd.DataFrame(x_RF, columns=data.columns)
# data_RF.to_excel('data/RF_fill.xlsx')

# #线性回归填补缺失值
# x_LR = inputer2.fit_transform(data)
# data_LR = pd.DataFrame(x_LR, columns=data.columns)
# data_LR.to_excel('data/LR_fill.xlsx')

# #K Nearest Neighbors  KNN填补缺失值
# x_KNN = inputer3.fit_transform(data)
# data_KNN = pd.DataFrame(x_KNN, columns=data.columns)
# data_KNN.to_excel('data/KNN(n=3)fill.xlsx')

#softImpute填补缺失值
data1 = data.to_numpy()
mid = BiScaler().fit_transform(data1)
x_SOFT = inputer4.fit_transform(mid)
data_SOFT = pd.DataFrame(x_SOFT, columns=data.columns)
data_SOFT.to_excel('data/SOFT填充.xlsx')
