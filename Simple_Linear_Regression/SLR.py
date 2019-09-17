import SimpleLinearRegression as model
import pandas as pd

dataset = pd.read_csv("insurance.csv")
dataset.columns = ['X','y']
size = 0.80
train = dataset[:int(len(dataset)*size)]
test = dataset[int(len(dataset)*size):]
y_pred = model.SimpleLinearRegression(train,test)
print(y_pred)
