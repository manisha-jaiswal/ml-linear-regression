import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statistics as m

arr1=np.array([1,2,3,4,5])
arr2=np.array([96,84,70,58,52])

x_=m.mean(arr1)
y_=m.mean(arr2)

arr3=arr1-x_
arr4=arr2-y_
arr5=arr3*arr4
s1=sum(arr5)
arr6=arr3*arr3
s2=sum(arr6)


m=s1/s2
c=y_-m*x_

x=10*np.random.rand(100)
y=m*x+c
plt.scatter(x,y)
plt.show()

#choose class of model
model=LinearRegression(fit_intercept=True)

#arrange data in matrix
x=x.reshape(-1,1)


#fit model
model.fit(x,y)
print(model.coef_,model.intercept_)


#data preparation for prediction
x_fit=np.linspace(-1,11)
x_fit=x_fit.reshape(-1,1)

#prediction
y_fit=model.predict(x_fit)
plt.scatter(x,y)
plt.plot(x_fit,y_fit)
plt.show()


