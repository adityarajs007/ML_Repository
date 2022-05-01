import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
df = pd.read_csv('canada_per_capita_income.csv', delimiter='\t')
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])
plt.xlabel("year")
plt.ylabel("Wages in $")
plt.plot(df['per capita income (US$)'],df.year*reg.coef_+reg.intercept_)
plt.show()
predicted_wage=reg.predict([[2020]])
print(predicted_wage)


