import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("practice_data/Salary_dataset.csv")
x = np.array(data.YearsExperience) 
y = np.array(data.Salary)

w = 0
b = 0

def linearReg(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m): 
        f_wb[i] = w * x[i] + b
    return f_wb

def costFun(x, y, w, b):
    m = x.shape[0]
    size = 1 / (2 * m)
    tmp_error = np.zeros(m)
    for i in range(m): 

        tmp_error[i] = (((w * x[i] + b) - y[i])**2)
    total = np.sum(tmp_error)
    mae = size * total
    return mae

def gradientDec(w, b, x, y, rate):
    m = len(x)
    for i in range(m):
        tmp_w = w - rate * (((w * x[i] + b) - y[i]) * x[i])
        tmp_b = b - rate * ((w * x[i] + b) - y[i])
        w = tmp_w
        b = tmp_b
    
    return w, b


def r_squared(y, y_hat):
    res_sum = np.sum((y - y_hat) ** 2)
    total_sum = np.sum((y - np.mean(y))  ** 2)
    cof_d = 1 - (res_sum/total_sum)
    return cof_d
    
learning_rate = 0.001
epoch = 1000
for i in range(epoch): 
    w, b = gradientDec(w,b,x,y,learning_rate)
    if i % 50 == 0: 
        print(f"Epoch: {i}: \t w: {w} \t b: {b}")
print(w,b)


y_hat = linearReg(x,w,b)
print(f"R^2: {r_squared(y, y_hat)}")


plt.plot(x, linearReg(x,w, b), c='b')
plt.xlabel("Years Expereince")
plt.ylabel("Salary($)")
plt.scatter(x, y, c='r' ,marker='x')
plt.show()



def model_version(x, y):
    print("-------")
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    
    new_sal = np.array([[69000]])
    y_pred = model.predict(x.reshape(-1, 1))
    predicted_price = model.predict(new_sal)
    
    mse = mean_squared_error(y, y_pred)
    print("Mean Squared Error:", mse)
    mae = mean_absolute_error(y, y_pred)
    print("Mean Absolute Error:", mae)
    print(r2_score(y,y_pred))
    print(predicted_price)

model_version(x,y)