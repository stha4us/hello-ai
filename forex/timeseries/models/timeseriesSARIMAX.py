import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import product
from dask import compute, delayed

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, pacf
import statsmodels.api as sm


class Sarimax_TimeSeries(object):
    def _checkData(self, input_file):
        if type(input_file) == pd.DataFrame:
            input_file = input_file.astype('float')
            return input_file, input_file.index
        else:
            raise ValueError("DataFrame with date index is require !")

    def __init__(self, data, optimize=False, verbose=0, **kwargs):
        self.endog, self.index = self._checkData(data)

        self.verbose = verbose
        self.optimize = optimize

        if 'exogenous' in kwargs:
            self.exog, _ = self._checkData(kwargs['exogenous'])
        else:
            self.exog = None

        if 'trend' in kwargs:
            self.trend = kwargs['trend']
        else:
            self.trend = 'ct'

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

    def _get_paramater(self, s):

        p = range(0, s, 1)
        P = range(0, s, 1)
        Q = range(0, s, 1)

        parameters = product(p, P, Q)
        param_list = list(parameters)
        if self.verbose == 1:
            print("[INFO] Total Parameter to check: {} ".format(len(param_list)))

        return param_list

    @delayed
    def _get_aic(self, data, exog, order, seasonal_order):
        try:
            model = SARIMAX(endog=data, exog=exog, order=order, seasonal_order=seasonal_order).fit(disp=-1)
            return model.aic
        except:
            pass

    def _hyperparameter_optimize(self, parameters_list, d, s, q, data, exog):
        """
        Return dataframe with parameters, corresponding AIC and SSE

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
        """
        results = []
        D = d

        for param in parameters_list:
            aic = self._get_aic(data, exog, (param[0], d, q), (param[1], D, param[2], s))
            results.append([(param[0], d, q), (param[1], D, param[2], s), aic])

        result_df = pd.DataFrame(results)
        result_df.columns = ['(p,d,q)', '(P,D,Q,s)', 'AIC']
        result_df['AIC'] = compute(*result_df['AIC'])

        # Sort in ascending order, lower AIC is better
        result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

        return result_df[:1].values[0][0], result_df[:1].values[0][1]

    def _fit(self, val):
        if self.verbose == 1:
            print('[INFO] Working To Reject Ho')

        # Finding the d value
        d = 0
        while self._ADCF_test(val) is False:
            val = val.diff().dropna()
            d = d + 1

        return val, d

    def _model(self, lags, confidence, p, s):
        data = self.endog
        exog = self.exog

        self.stationary, d = self._fit(self.endog)
        q = self._get_lags(self.stationary, lags, confidence)

        if self.optimize:
            test_params = self._get_paramater(s)
            order, season = self._hyperparameter_optimize(test_params, d, s, q, data, exog)
        else:
            # Finding p , d and q
            # p = 5 #TODO: Get p dynamically using acf
            # Finding P,D,Q,s

            order = (p, d, q)
            P = 1
            D = d
            Q = 0
            s = s
            season = (P, D, Q, s)

        if self.verbose == 1:
            print('[INFO] Using Order {}x{}'.format(order, season))
        # final model
        model = SARIMAX(data, exog=exog, order=order, seasonal_order=season,
                        enforce_invertibility=self.enforce_invertibility,
                        trend_offset=self.trend_offset, trend=self.trend,
                        enforce_stationarity=self.enforce_stationarity)
        return model

    def fit(self, p=5, season=12):
        print('[INFO] Compiling Model ..')
        self.fitted_model = self._model(self.lags, self.alpha, p, season).fit()
        return self.fitted_model

    def predict(self, exo_test, start=None, end=None, dynamic=False):
        prediction = self.fitted_model.predict(start=start,
                                               end=end,
                                               dynamic=dynamic,
                                               exog=exo_test)

        return pd.DataFrame(prediction)

    def get_prediction(self, exo_test, start=None, end=None, dynamic=False):
        return self.fitted_model.get_prediction(start=start,
                                                end=end,
                                                dynamic=dynamic,
                                                exog=exo_test).predicted_mean

    def forecast(self, exo_test, steps=10):
        forecasted = self.fitted_model.forecast(steps=steps, exog=exo_test)

        return pd.DataFrame(forecasted)

    def get_forecast(self, steps, exo_test):
        return self.fitted_model.get_forecast(steps=steps,
                                              exog=exo_test).predicted_mean

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
        true, pred = np.array(true[0]), np.array(pred[:, 0])

        return abs(true - pred).sum() / abs(true).sum()

    @staticmethod
    def r_squared_adj(true, pred):
        # true, pred = np.array(true[0]), np.array(pred[:,0])
        true_addC = sm.add_constant(true)

        return (sm.OLS(pred, true_addC).fit()).rsquared_adj

    def accuracy(self):
        # make predictions
        test_predict = self.predict(exo_test=self.exog).values
        y_test = self.endog.values

        print('Mean Squared Error:', self.mean_squared_error(y_test, test_predict))
        print('Root Mean Squared Error:', self.RMS(y_test, test_predict))
        print('Mean Absolute Percentage Error:', self.mean_absolute_percentage_error(y_test, test_predict))
        print("AIC: ", self.fitted_model.aic)

        return {'Mean_Squared_Error': self.mean_squared_error(y_test, test_predict),
                'Root_Mean_Error': self.RMS(y_test, test_predict),
                'Mean_Absolute_Error': self.mean_absolute_percentage_error(y_test, test_predict),
                'WMAE': self.weighted_mean_absolute_error(y_test, test_predict),
                'R2': self.r_squared_adj(y_test, test_predict)
                }