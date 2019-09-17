# Simple Linear Regression from scratch

def SimpleLinearRegression(train, test):
    predictions = []
    b0, b1 = coefficients(train)
    for val in test['X']:
        yhat = b0+b1*val
        predictions.append(yhat)
    return predictions

def coefficients(train):
    x = list(train['X'])
    y = list(train['y'])
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1*x_mean
    return [b0,b1]

def mean(values):
    return sum(values) / float(len(values))

def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i]-mean_x) * (y[i]-mean_y)
    return covar
