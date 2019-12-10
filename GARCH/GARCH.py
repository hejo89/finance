"""The code generates and plots a time series with conditional heteroscedasticity as well as the
squared time series. The respective autocorrelation and partial autocorrelation functions are 
plotted as well."""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import pacf

def plot_acf(arr, number_of_lags):
    lags = []
    for l in range(number_of_lags + 1):
        m = sum(arr) / len(arr)
        result = [(arr[i] - m)  * (arr[i + l] - m) for i in range(len(arr) - l)]
        lags.append(sum(result)/sum([(k - m)**2 for k in arr])*len(arr)/(len(arr)-l))

    plt.axhline(y = 1.96 / pow(len(arr), 0.5), linewidth=2, color='r')
    plt.axhline(y = -1.96 / pow(len(arr), 0.5), linewidth=2, color='r')
    plt.axhline(color='black')
    plt.plot(range(len(lags)), lags, "bo")
    for i in range(len(lags)):
        plt.vlines(x=i, ymin = min(0, lags[i]), ymax = max(0, lags[i]))

    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title('ACF')

    plt.show()

def plot_pacf(arr, number_of_lags):
    lags = pacf(arr, nlags=number_of_lags)
    plt.axhline(y = 1.96 / pow(len(arr), 0.5), linewidth=2, color='r')
    plt.axhline(y = -1.96 / pow(len(arr), 0.5), linewidth=2, color='r')
    plt.axhline(color='black')
    plt.plot(range(len(lags)), lags, "bo")
    for i in range(len(lags)):
        plt.vlines(x=i, ymin = min(0, lags[i]), ymax = max(0, lags[i]))

    plt.xlabel('Lags')
    plt.ylabel('Partial autocorrelation')
    plt.title('PACF')

    plt.show()

X, sigma_squared = [0], [0]
alpha_0, alpha_1 = 0.5, 0.5
beta_1 = 0.5
for i in range(1000):
    sigma_squared.append(alpha_0 + alpha_1 * X[i]**2 + beta_1 * sigma_squared[i])
    X.append(pow(sigma_squared[-1], 0.5) * np.random.normal())

plt.plot(range(len(X)), X)
plt.show()
plot_acf(X, 30)
plot_pacf(X, 30)

X = [k**2 for k in X]
plt.plot(range(len(X)), X)
plt.show()
plot_acf(X, 30)
plot_pacf(X, 30)
