import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
# loading data from csv file to pandas data frame
df=pd.read_csv("houseprices.csv")
print(df)

# plot a scatter plot for getting the distribution of data points
plt.scatter(df.area,df.price,color ="red", marker='+')
plt.xlabel("area (in sqft)")
plt.ylabel("price (in USD)")
#plt.show()

# using the linear regression model.
# step 1: create a linear regression object

reg= linear_model.LinearRegression()

# Step 2: fitting the model- means to use the technique to generate a function
# that minimises the error value or deviations from each data point
reg.fit(df[['area']],df.price) # if no error at this line, then the model is ready to predict
'''
predicted_price=float(reg.predict([[3300]]))
print('predicted_price',predicted_price)
# using y=mx+c, m=coeff, x= area, c= intercept, y= house price
print('coeff',reg.coef_)
print('intercept',reg.intercept_)
plt.plot(df.area,df.area*reg.coef_+reg.intercept_) # same as plt(x, m*x+c)
plt.show()
'''
# to predict many prices for different area givven in csv file

# create a panda data frame

df1=pd.read_csv("houseareas.csv")
#array=df1
#print(array)
#print(df1.area)
predicted_price_list=reg.predict(df1)
#plt.scatter(df1,predicted_price_list,color ="red", marker='*')
#plt.xlabel("area (in sqft)")
#plt.ylabel("price (in USD)")
#plt.show()
df1['prices']=predicted_price_list # adds a new data frame column called 'prices'
df1.to_csv("prediction.csv") # prints the data frame 'area' and' prices' into a new csv file