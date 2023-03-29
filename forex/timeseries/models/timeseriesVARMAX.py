# VARMAX example
import json
import warnings
import copy
warnings.simplefilter("ignore")
import numpy as np
from math import floor
import pandas as pd
from scipy.stats import mstats
import statsmodels.tsa as ts
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error
from itertools import product
from dask import compute, delayed
from statsmodels.tsa.stattools import adfuller, pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests



class TimeseriesVARMAX(object):
    def _checkData(self, input_file):
        if type(input_file) == pd.DataFrame:
            input_file = input_file.astype('float')
            return input_file, input_file.index
        else:
            raise ValueError("DataFrame with date index is require !")

    def __init__(self, data, target_column, endog_columns, exog_columns, optimize=False, verbose=0, **kwargs):
        self.processed_traindata_dimension = (0,0)
        self.verbose = verbose
        self.optimize = optimize
        self.target_column = target_column
        self.endog_columns = endog_columns
        self.exog_columns = exog_columns
        self.data, self.index = self._checkData(data)
        self.fit_model, self.lag_order= False, False
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
            self.lags = 20

        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 0.05

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, param, deep=True):
        return getattr(self, param, None)

    # result = adfuller(y)
    # print("ADF Statistics", result[0])
    # print("P value", result[1])

    # print('Results of Dickey Fuller Test:')
    # dftest = adfuller(forcast['T2M_MAX'], autolag='AIC')

    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    # for key,value in dftest[4].items():
    #     dfoutput['Critical Value (%s)'%key] = value

    # print(dfoutput)

    def _ADCF_test(self, X):
        try:
            x = X.values
        except:
            x = X
        result = adfuller(x, autolag='AIC')

        if self.verbose is 1:
            print('\tLag Used: %i' % result[2])
            print('\tADF Statistic: %f' % result[0])
            print('\tp-value: %f' % result[1])
            print('\tNumber of Observation: %i' % result[3])
            print('\tCritical Values:')

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

    def _get_paramater(self):

        p = range(0, 1)
        P = range(0, 1)
        Q = range(0, 1)

        parameters = product(p, P, Q)
        param_list = list(parameters)
        if self.verbose == 1:
            print("[INFO] Total Parameter to check: {} ".format(len(param_list)))

        return param_list

    @delayed
    def _get_aic(self, data, exog, order):
        try:
            model = VARMAX(endog=data, exog=exog, order=order).fit(disp=-1)
            return model.aic
        except:
            pass

    def _hyperparameter_optimize(self, parameters_list, q, data, exog):
        """
        Return dataframe with parameters, corresponding AIC and SSE

        parameters_list - list with (p, q, P, Q) tuples
        exog - the exogenous variable
        """
        results = []
        # D = d

        for param in parameters_list:
            aic = self._get_aic(data, exog, (param[0], q))
            results.append([(param[0], q), (param[1], param[2]), aic])

        result_df = pd.DataFrame(results)
        result_df.columns = ['(p,q)', '(P,Q)', 'AIC']
        result_df['AIC'] = compute(*result_df['AIC'])

        # Sort in ascending order, lower AIC is better
        result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

        return result_df[:1].values[0][0], result_df[:1].values[0][1]

    def _fit(self, val):
        #@TODO: changes
        y = copy.deepcopy(val)
        val = y[self.target_column].resample('MS').mean()
        if self.verbose == 1:
            print('[INFO] Working To Reject Ho')

        # Finding the d value
        d = 0
        while self._ADCF_test(val) is False:
            val = val.diff().dropna()
            d = d + 1

        return val, d

    def _model(self, lags, confidence, p, **kwargs):
        # if 'tranformation' in kwargs:
        #     data[self.target_column+'_log'] = np.log(forcast["T2M_MAX"])
        # self.stationary, _ = self._fit(self.endog)
        # q = self._get_lags(self.stationary, lags, confidence)
        val = self.endog[self.target_column]
        q = self._get_lags(val, lags, confidence)

        if self.optimize:
            test_params = self._get_paramater()
            order, season = self._hyperparameter_optimize(test_params, q, self.endog, self.exog)
        else:
            order = (p, q)

        if self.verbose == 1:
            print('[INFO] Using Order {}'.format(order))
        model = VARMAX(self.endog.values, exog=self.exog, order=order,
                       error_cov_type='unstructured', measurement_error=False,
                       enforce_invertibility=self.enforce_invertibility,
                       trend_offset=self.trend_offset, trend=self.trend,
                       enforce_stationarity=self.enforce_stationarity)
        return model

    def fit_old_method(self, p=5):
        print('[INFO] Compiling Model ..')
        self.fitted_model = self._model(self.lags, self.alpha, p, tranformation=True).fit()
        return self.fitted_model

    def fit(self, **kwargs):
        print('[INFO] Compiling Model ..')
        storeTestData, TestData = self.train_test_split(self.data, self.exog_columns)
        self.processed_traindata_dimension = storeTestData.shape

        endog = storeTestData[self.endog_columns].astype('float32')
        exog = storeTestData[self.exog_columns].astype('float32')

        if (kwargs['time_shifting'] if 'time_shifting' in kwargs else None):
            endog, exog, rank = self.difference_series(endog, exog)
        self.fit_model, self.lag_order = self.model_generator(endog, exog, verbose=1)
        return endog, exog

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

    def forecast_old(self, exo_test, steps=10):
        forecasted = self.fitted_model.forecast(steps=steps, exog=exo_test)
        return pd.DataFrame(forecasted)

    def forecast(self, endog, future_exog=False, steps=14, verbose=0, **kwargs):
        model = self.fit_model.fit(maxiter=kwargs['maxiterations'] if 'maxiterations' in kwargs else 100, disp=False)
        if verbose == 1:
            print(model.summary())
            print(model.params)
        if self.lag_order:
            forecasted = pd.DataFrame(self.__forecast(model, endog, future_exog, steps=steps, lag_order=self.lag_order))
        else:
            forecasted = pd.DataFrame(self.__forecast(model, endog, future_exog, steps=steps))
        actual, predicted = self.actual_predicted_data(endog, forecasted, steps=steps)

        pred = pd.merge(actual, predicted, right_index=True, left_index=True)
        pred = pred[pred['Actual' + self.target_column] != 0]
        pred.columns = ['Actual' + self.target_column, self.target_column]
        if 'get_json_data' in kwargs:
            result = pred.to_json(orient="index")
            parsed_result = json.loads(result)
            return pred, predicted, parsed_result
        return pred, predicted

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
        true, pred = np.array(true[:, 0]), np.array(pred[:, 0])

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
        print("Weighted Mean absolute error:",self.weighted_mean_absolute_error(y_test, test_predict))

        return {'Mean_Squared_Error': self.mean_squared_error(y_test, test_predict),
                'Root_Mean_Error': self.RMS(y_test, test_predict),
                'Mean_Absolute_Percentage_Error': self.mean_absolute_percentage_error(y_test, test_predict),
                'WMAE': self.weighted_mean_absolute_error(y_test, test_predict),
                'R2': self.r_squared_adj(y_test, test_predict),}

    # get transformed values
    def data_transformer(self, data, to_transfer_columns, limits=[0.05, 0.05])->'dataframe':
        """
        data: pandas dataframe
        columns: subset of list of columns in data
        limits: threshold limit range for data transformation
        """
        for cols in to_transfer_columns:
            transformed_data = pd.Series(mstats.winsorize(data[str(cols)], limits=limits))
            data[str(cols)] = transformed_data.values
        return data

    ################# get seasonality and trend ##########################
    ##### change codde to include search for best sesonality using freq#############
    def seasonal_decomposer(self, data, endog_columns , model='additive', freq=365, extrapolate_trend='freq')->'result':
        for cols in endog_columns:
            result = seasonal_decompose((data[cols]), model=model, freq=freq, extrapolate_trend=extrapolate_trend)
            data[cols + "Seasonality"] = result.seasonal
            data[cols + "Trend"] = result.trend
            return data, result

    def train_test_split(self, data, exog_columns, test_fraction=0.2, **kwargs):
        test_data = floor(test_fraction * len(data))
        trainData, testData = data[:-test_data], data[-test_data:]
        if 'causality_test' in kwargs:
            if kwargs['causality_test']:
                first_list = [self.target_column]
                first_list.extend(list(set(exog_columns) - set([self.target_column])))
                second_list = list(set(exog_columns) - set([self.target_column]))
                second_list.extend([self.target_column])
                CausalitySales = (ts.stattools.grangercausalitytests(trainData[first_list].dropna(), 1))
                CausalityCust = (ts.stattools.grangercausalitytests(trainData[second_list].dropna(), 1))
        return trainData, testData


    def addfuller_test(self, train, secondary_column):
        station = adfuller(train[self.target_column], autolag='AIC')
        if station[4]['5%'] < station[0]:
            stationDIF1 = adfuller(train[secondary_column].diff().dropna(), autolag='AIC')
        station = adfuller(train[secondary_column], autolag='AIC')
        if station[4]['5%'] < station[0]:
            stationDIF2 = adfuller(train[secondary_column].diff().dropna(), autolag='AIC')
        return stationDIF1, stationDIF2

    def difference_series(self, endog, exog):
        endogdif1 = endog.diff().dropna()
        exogdif1 = exog.diff().dropna()

        endogdif11 = endog.iloc[:, 0].diff().dropna()
        endogdif1.iloc[:, 0] = endogdif11.values

        endogdif12 = endog.iloc[:, 1].diff().dropna()
        endogdif1.iloc[:, 1] = endogdif12.values

        coint = coint_johansen(endogdif1, 0, 1)
        traces = coint.lr1
        maxeig = coint.lr2
        cvts = coint.cvt  ## 0: 90%  1:95% 2: 99%
        cvms = coint.cvm
        N, l = endogdif1.shape

        for i in range(l):
            if traces[i] > cvts[i, 1]:
                r = i + 1

        rank = select_coint_rank(endogdif1, 0, 1)
        return endogdif1, exogdif1, rank

    def model_generator(self,endogdif1, exogdif1, verbose=0, threshold_records=10000, **kwargs):
        if (kwargs['var_model_value'] if 'var_model_value' in kwargs else False):
            mod = VAR(endogdif1, exog=exogdif1)  # , order=(2,0,0)
            aa = mod.select_order()
            res = mod.fit(maxlags=aa.aic, ic='aic')
            lag_order = res.k_ar
            if verbose == 1:
                print(aa.aic)
                print(res.summary())
        else:
            lag_order=None
        p_val = 6 if self.processed_traindata_dimension[0] > threshold_records else 3
        q_val = 3 if self.processed_traindata_dimension[0] > threshold_records else 1

        model = VARMAX(endogdif1, exog=exogdif1, order=(p_val, q_val), trend='n')  # endogdif1, exogdif1
        return model, lag_order


    def __impulse_response(self, model):
        return model.impulse_response(steps=100, orthogonalized=False)


    def __forecast(self, model, endog, exog, steps = 14, **kwargs):
        if 'only_forecast' in kwargs:
            if 'lag_order' in kwargs:
                forecast_input = endog.values[-kwargs['lag_order']:]
                forcast = model.forecast(steps=steps, y=forecast_input, exog=exog)
            else:
                forcast = model.forecast(steps=steps, exog=exog)
        else:
            if 'lag_order' in kwargs:
                forecast_input = endog.values[-kwargs['lag_order']:]
                forcast = model.predict(steps=len(endog)+steps, y=forecast_input, exog=exog)
            else:
                forcast = model.predict(steps=len(endog)+steps, exog=exog)
        return forcast


    def actual_predicted_data(self, endogdif1, forecast, steps=14, **kwargs):
        #@TODO: need to consider the steps to filter actual predicted filtering
        actual = endogdif1[[self.target_column]]
        actual.columns = ['Actual'+self.target_column]
        actual = actual.reset_index(drop=True)
        # predicted = forcast.iloc[:, 0]
        predicted = forecast.reset_index(drop=True)
        return actual, predicted


    def accuracy(self, actual, predicted, verbose=0):
        if round(actual/1.0,1)==0.0:
            denom = round(actual.mean(),2)
        # dinom[dinom < 0] = abs(dinom[[dinom < 0]])
        MPE = np.mean((actual - predicted) / (denom))
        MAPE = np.mean(abs(actual - predicted) / (denom))
        accuracy = 100-(MAPE*100)
        if verbose == 1:
            print('MPE: ', MPE*100)
            print('MAPE: ', MAPE*100)
            print('Accuracy: ', 100-(MAPE*100))
        return MPE, MAPE, accuracy



    #ANALYTICS PART!
    def causality_test(self, df):
        granger_test = sm.tsa.stattools.grangercausalitytests(df, maxlag=2, verbose=True)
        return granger_test

    def train_test_split(self,df, nobs=10):
        df_train, df_test = df[0:-nobs], df[-nobs:]
        return df_train, df_test


    def adf_test(ts, signif=0.05):
        dftest = adfuller(ts, autolag='AIC')
        adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# Lags', '# Observations'])
        for key, value in dftest[4].items():
            adf['Critical Value (%s)' % key] = value
        print(adf)

        p = adf['p-value']
        if p <= signif:
            print(f" Series is Stationary")
        else:
            print(f" Series is Non-Stationary")

    # inverting transformation from the so made stationary dataset
    def invert_transformation(self, df_train, df_forecast, second_diff=False):
        """Revert back the differencing to get the forecast to original scale."""
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:
            # Roll back 2nd Diff
            if second_diff:
                df_fc[str(col) + '_1d'] = (df_train[col].iloc[-1] - df_train[col].iloc[-2]) + df_fc[
                    str(col) + '_1d'].cumsum()
            # Roll back 1st Diff
            df_fc[str(col) + '_forecast'] = df_train[col].iloc[-1] + df_fc[str(col) + '_1d'].cumsum()
        return df_fc

    def overall_flow(self,df, nobs=4):
        df_train, df_test = self.train_test_split()

        # apply adf test on the series
        self.adf_test(df_train["realgdp"])
        self.adf_test(df_train["realcons"])

        # 1st difference
        df_differenced = df_train.diff().dropna()
        # stationarity test again with differenced data
        self.adf_test(df_differenced["realgdp"])

        # model fitting
        model = VAR(df_differenced)
        results = model.fit(maxlags=15, ic='aic')
        results.summary()

        # forecasting
        lag_order = results.k_ar
        results.forecast(df.values[-lag_order:], 5)
        # plotting
        results.plot_forecast(20)

        # Evaluation using Forecast Error Variance Decomposition
        fevd = results.fevd(5)
        fevd.summary()

        # forecasting
        pred = results.forecast(results.y, steps=nobs)
        df_forecast = pd.DataFrame(pred, index=df.index[-nobs:], columns=df.columns + '_1d')
        df_forecast.tail()

        # show inverted results in a dataframe
        df_results = self.invert_transformation(df_train, df_forecast, second_diff=True)
        df_results.loc[:, ['realgdp_forecast', 'realcons_forecast']]





