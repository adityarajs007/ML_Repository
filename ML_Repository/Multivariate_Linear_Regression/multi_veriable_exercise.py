import pandas as pd
from sklearn import linear_model

df=pd.read_csv('hiring.csv')
median= df['test_score(out of 10)'].median()
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(median)
df['experience']=df['experience'].fillna(0)
reg=linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
print(reg.predict([[2,9,6]]))
print(reg.predict([[12,10,10]]))