import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, pacf
import statsmodels.api as sm

class Arima_TimeSeries(object):
    def _checkData(self, input_file):
        if type(input_file) == pd.DataFrame:
            input_file = input_file.astype('float')
            return input_file, input_file.index
        else:
            raise ValueError("DataFrame with date index is required !")

    def __init__(self, data, verbose=0, **kwargs):
        self.endog, self.index = self._checkData(data)
        self.verbose = verbose

        if 'exog' in kwargs:
            self.exog = kwargs['exog']
        else:
            self.exog = None

        if 'trend' in kwargs:
            self.trend = kwargs['trend']
        else:
            self.trend = 't'

        if 'enforce_invertibility' in kwargs:
            self.enforce_invertibility = kwargs['enforce_invertibility']
        else:
            self.enforce_invertibility = True

        if 'enforce_stationarity' in kwargs:
            self.enforce_stationarity = kwargs['enforce_stationarity']
        else:
            self.enforce_stationarity = True

        if 'trend_offset' in kwargs:
            self.trend_offset = kwargs['trend_offset']
        else:
            self.trend_offset = 1

        if 'missing' in kwargs:
            self.missing = kwargs['missing']
        else:
            self.missing = 'drop'

        if 'lags' in kwargs:
            self.lags = kwargs['lags']
        else:
            self.lags = 40

        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 0.05

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        pass

    def _ADCF_test(self, X):
        try:
            x = X.values
        except:
            x = X
        result = adfuller(x)

        if self.verbose is 1:
            print('\tLag Used: %i' % result[2])
            print('\tADF Statistic: %f' % result[0])
            print('\tp-value: %f' % result[1])
            print('\tNumber of Observation: %i' % result[3])
            print('\tCritical Values:')

            # self._plot_rolling(X,window_size=window)
            for key, value in result[4].items():
                print('\t%s: %.3f' % (key, value))

        if result[0] < result[4]["5%"]:

            print("[INFO] Reject Ho - Time Series is Stationary")
            decision = True

        else:
            print("[WARNING] Failed to Reject Ho - Time Series is Non-Stationary")
            print("[INFO] Transformation for making Stationary")
            decision = False

        return decision

    def _get_lags(self, dataset, nlags, alpha, method='ld'):
        _, confidence = pacf(dataset, nlags=nlags, method=method, alpha=alpha)
        return (confidence >= 1).sum()

    def _log_conversion(self, x):
        result = np.log(x)
        return result.replace([np.inf, -np.inf], 0).dropna(axis=1)

    def _exponential_conversion(self, x):
        return np.exp(x)

    def _fit(self, val):
        if self.verbose == 1:
            print('[INFO] Working To Reject Ho')
        decision = self._ADCF_test(val)
        if decision:
            return None, val, 0
        else:
            log_transform = self._log_conversion(val)
            decision = self._ADCF_test(log_transform)
            if decision:
                return 'LOG', log_transform, 1
            else:
                print("[WARNING] Data is not stationary")
                return 'LOG', log_transform, 2

    def _model(self, lags, confidence, p):
        # Getting data
        self.convert, data, d = self._fit(self.endog)
        # Finding p , d and q
        # p = 5 #TODO: Get p dynamically using acf
        q = self._get_lags(data, lags, confidence)
        order = (p, d, q)
        if self.verbose == 1:
            print('[INFO] Using Order {}'.format(order))
        # final model
        model = ARIMA(data, order=order, missing=self.missing,
                      enforce_invertibility=self.enforce_invertibility,
                      trend_offset=self.trend_offset, trend=self.trend,
                      enforce_stationarity=self.enforce_stationarity)
        return model

    def fit(self, p=5):
        print('[INFO] Compiling Model ..')
        self.fitted_model = self._model(self.lags, self.alpha, p).fit()
        return self.fitted_model

    def predict(self, start=None, end=None, dynamic=False):
        prediction = self.fitted_model.predict(start, end, dynamic)

        if self.convert == 'LOG':
            prediction = self._exponential_conversion(prediction)

        return pd.DataFrame(prediction)

    def get_prediction(self, start=None, end=None, dynamic=False):
        return self.fitted_model.get_prediction(start, end, dynamic)

    def forecast(self, steps=10):
        forecasted = self.fitted_model.forecast(steps)
        if self.convert == 'LOG':
            forecasted = self._exponential_conversion(forecasted)

        return pd.DataFrame(forecasted)

    def get_forecast(self, steps):
        return self.fitted_model.get_forecast(steps)

    @staticmethod
    def mean_squared_error(X, Y):
        return mean_squared_error(X, Y)

    def RMS(self, X, Y):
        return np.sqrt(self.mean_squared_error(X, Y))

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        dinom = y_true
        # changing denominator zero to 1
        # using this to change only denominator to skip divide by zero error
        # eventually zero values will result to be zero
        dinom[dinom == 0] = dinom.mean()
        # dinom[dinom < 0] = abs(dinom[[dinom < 0]])
        return np.abs((y_true - y_pred) / (dinom)).mean()

    @staticmethod
    def weighted_mean_absolute_error(true, pred):
        true, pred = np.array(true[0]), np.array(pred[:,0])

        return abs(true - pred).sum() / abs(true).sum()

    @staticmethod
    def r_squared_adj(true, pred):
        true_addC = sm.add_constant(true)

        return (sm.OLS(pred, true_addC).fit()).rsquared_adj

    def accuracy(self):
        # make predictions
        test_predict = self.predict().values
        y_test = self.endog.values

        print('Mean Squared Error:', self.mean_squared_error(y_test, test_predict))
        print('Root Mean Squared Error:', self.RMS(y_test, test_predict))
        print('Mean Absolute Percentage Error:',self.mean_absolute_percentage_error(y_test,test_predict))

        return {'Mean_Squared_Error': self.mean_squared_error(y_test, test_predict),
                'Root_Mean_Error': self.RMS(y_test, test_predict),
                'Mean_Absolute_Error': self.mean_absolute_percentage_error(y_test, test_predict),
                'WMAE':self.weighted_mean_absolute_error(y_test,test_predict),
                'R2': self.r_squared_adj(y_test, test_predict)
                }

