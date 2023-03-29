from utils import helpers as hp
import click
from .models.timeseriesLSTM import Lstm_univarient as LSTM_MODEL_UNIVARIENT
from .models.timeseriesARIMA import Arima_TimeSeries
from .models.timeseriesSARIMA import Sarima_TimeSeries
from .models.timeseriesSARIMAX import Sarimax_TimeSeries
from .models.timeseriesVARMAX import TimeseriesVARMAX

class Helpers(object):
    def __init__(self):
        self.forecast = []
        self.nn_accuracy = []
        self.total_data_explored = []
        self.model_name = []
        self.r2 = False

    def handle_timeseriesLSTM(self, config, processed_data_path,user_directory, **kwargs ):
        activation_function = kwargs['activation_function']
        batch_size = kwargs['batch_size']
        model_optimizer = kwargs['model_optimizer']
        model_loss = kwargs['model_loss']
        model_metrics = kwargs['model_metrics']
        dropout_rate = kwargs['dropout_rate']

        n_neuron = config['n_neuron']
        look_back = config['look_back']

        to_data = int(config['n_forecast']) if 'n_forecast' in config else 14

        timeseries_processed_data = hp.read_pickle(processed_data_path)
        total_data_explored = len(timeseries_processed_data.index)

        model = LSTM_MODEL_UNIVARIENT(user_id=user_directory, path=processed_data_path,
                                      activation_function=activation_function, batch_size=batch_size,
                                      model_optimizer=model_optimizer, \
                                      model_loss=model_loss, model_metrics=model_metrics,
                                      dropout_rate=dropout_rate, epochs_number=30, look_back=look_back,
                                      n_neuron=n_neuron)

        model.prepare_fit(timeseries_processed_data)
        forecast = model.forecast(to_data)
        accuracies = model.accuracy()
        r2_accuracy = accuracies['R2']
        mape_accuracy = accuracies['Mean_Absolute_Error']
        print("Accuracies",r2_accuracy," : ",mape_accuracy)
        if abs(100-mape_accuracy*100) > 100:  nn_accuracy = r2_accuracy * 100; self.r2=True
        else: nn_accuracy = 100 - mape_accuracy * 100

        self.forecast.append(forecast)
        self.nn_accuracy.append(nn_accuracy)
        self.total_data_explored.append(total_data_explored)
        self.model_name.append("LSTM")

    def handle_timeseriesARIMA(self, config, processed_data_path, **kwargs ):
        n_neuron = config['n_neuron']
        look_back = config['look_back']

        to_data = int(config['n_forecast']) if 'n_forecast' in config else 14

        timeseries_processed_data = hp.read_pickle(processed_data_path)
        total_data_explored = len(timeseries_processed_data.index)
        try:
            exog = config['exog']
            trend = config['trend']
            enforce_invertibility = config['enforce_invertibility']
            enforce_stationarity = config['enforce_stationarity']
            trend_offset = config['trend_offset']
            missing = config['missing']
            lags = config['lags']
            alpha = config['alpha']
            model = Arima_TimeSeries(timeseries_processed_data,
                                          exog=exog, trend=trend,
                                          enforce_invertibility=enforce_invertibility, \
                                          enforce_stationarity=enforce_stationarity, trend_offset=trend_offset,
                                          missing=missing, lags=lags, alpha=alpha, epochs_number=30, look_back=look_back,
                                          n_neuron=n_neuron)
        except:
            model = Arima_TimeSeries(timeseries_processed_data,\
                                     epochs_number=30, look_back=look_back,n_neuron=n_neuron)

        model.fit()
        # pred = model.predict()
        # get_pred = model.get_prediction()
        # get_forc = model.get_forecast(steps=to_data)
        forecast = model.forecast(steps= to_data)
        accuracies = model.accuracy()
        r2_accuracy = accuracies['R2']
        mape_accuracy = accuracies['Mean_Absolute_Error']
        if abs(100-mape_accuracy*100) > 100:  nn_accuracy = r2_accuracy * 100; self.r2=True
        else: nn_accuracy = 100 - mape_accuracy * 100

        # array=[pred,get_pred,get_forc]
        self.forecast.append(forecast)
        self.nn_accuracy.append(nn_accuracy)
        self.total_data_explored.append(total_data_explored)
        self.model_name.append("ARIMA")



    def handle_timeseriesSARIMA(self,config, processed_data_path, **kwargs ):
        n_neuron = config['n_neuron']
        look_back = config['look_back']

        to_data = int(config['n_forecast']) if 'n_forecast' in config else 14

        timeseries_processed_data = hp.read_pickle(processed_data_path)
        total_data_explored = len(timeseries_processed_data.index)
        try:
            exog = config['exog']
            trend = config['trend']
            enforce_invertibility = config['enforce_invertibility']
            enforce_stationarity = config['enforce_stationarity']
            trend_offset = config['trend_offset']
            missing = config['missing']
            lags = config['lags']
            alpha = config['alpha']
            model = Sarima_TimeSeries(timeseries_processed_data, exog=exog, trend=trend,
                                     enforce_invertibility=enforce_invertibility, \
                                     enforce_stationarity=enforce_stationarity, trend_offset=trend_offset,
                                     missing=missing, lags=lags, alpha=alpha, epochs_number=30, look_back=look_back,
                                     n_neuron=n_neuron)
        except:
            model = Sarima_TimeSeries(timeseries_processed_data, \
                                     epochs_number=30, look_back=look_back, n_neuron=n_neuron)

        model.fit()
        forecast = model.forecast(steps=to_data)
        accuracies = model.accuracy()
        r2_accuracy = accuracies['R2']
        mape_accuracy = accuracies['Mean_Absolute_Error']

        if abs(100-mape_accuracy*100) > 100:  nn_accuracy = r2_accuracy * 100; self.r2=True
        else: nn_accuracy = 100 - mape_accuracy * 100

        self.forecast.append(forecast)
        self.nn_accuracy.append(nn_accuracy)
        self.total_data_explored.append(total_data_explored)
        self.model_name.append("SARIMA")

    def handle_timeseriesSARIMAX(self,config, processed_data_path, **kwargs ):
        n_neuron = config['n_neuron']
        look_back = config['look_back']

        to_data = int(config['n_forecast']) if 'n_forecast' in config else 14

        timeseries_processed_data = hp.read_pickle(processed_data_path)
        total_data_explored = len(timeseries_processed_data.index)
        try:
            exog = config['exog']
            trend = config['trend']
            enforce_invertibility = config['enforce_invertibility']
            enforce_stationarity = config['enforce_stationarity']
            trend_offset = config['trend_offset']
            missing = config['missing']
            lags = config['lags']
            alpha = config['alpha']
            model = Sarimax_TimeSeries(timeseries_processed_data, exogenous=exog, trend=trend,
                                     enforce_invertibility=enforce_invertibility, \
                                     enforce_stationarity=enforce_stationarity, trend_offset=trend_offset,
                                     missing=missing, lags=lags, alpha=alpha, epochs_number=30, look_back=look_back,
                                     n_neuron=n_neuron)
            model.fit()

            ## For forecast we require test exogenous data
            forecast = model.forecast(exog[-to_data:],steps=to_data)

        except:
            model = Sarimax_TimeSeries(timeseries_processed_data, \
                                     epochs_number=30, look_back=look_back, n_neuron=n_neuron)

            model.fit()
            forecast = model.forecast(steps=to_data)

        accuracies = model.accuracy()
        r2_accuracy = accuracies['R2']
        mape_accuracy = accuracies['Mean_Absolute_Error']

        if abs(100-mape_accuracy*100) > 100:  nn_accuracy = r2_accuracy * 100; self.r2=True
        else: nn_accuracy = 100 - mape_accuracy * 100

        self.forecast.append(forecast)
        self.nn_accuracy.append(nn_accuracy)
        self.total_data_explored.append(total_data_explored)
        self.model_name.append("SARIMA")


    def handle_timeseriesVARMAX(self,config, processed_data_path, **kwargs ):
        n_neuron = config['n_neuron']
        look_back = config['look_back']

        to_data = int(config['n_forecast']) if 'n_forecast' in config else 14

        timeseries_processed_data = hp.read_pickle(processed_data_path)
        total_data_explored = len(timeseries_processed_data.index)
        try:
            trend = config['trend']
            enforce_invertibility = config['enforce_invertibility']
            enforce_stationarity = config['enforce_stationarity']
            trend_offset = config['trend_offset']
            missing = config['missing']
            lags = config['lags']
            alpha = config['alpha']
            data_endog_columns = config['endog_columns']
            data_exog_columns = config['exog_columns']
            target_column = config['exog_columns']
            model = TimeseriesVARMAX(timeseries_processed_data, target_column, data_endog_columns, data_exog_columns, verbose=0,
                                     trend=trend, enforce_invertibility=enforce_invertibility,
                                     enforce_stationarity=enforce_stationarity, trend_offset=trend_offset, \
                                     missing=missing, lags=lags, alpha=alpha, epochs_number=30, look_back=look_back,\
                                     n_neuron=n_neuron)

        except:
            data_endog_columns = config['endog_columns']
            data_exog_columns = config['exog_columns']
            target_column = config['exog_columns']
            model = TimeseriesVARMAX(timeseries_processed_data, target_column, data_endog_columns, data_exog_columns, verbose=0)

        endog, _ =model.fit()
        pred, forecast = model.forecast(endog,steps=to_data, verbose=0, future_exog=False, **kwargs)
        accuracies = model.accuracy(pred['Actual'+self.target_column], pred[self.target_column])
        r2_accuracy = accuracies['R2']
        mape_accuracy = accuracies['Mean_Absolute_Error']
        #
        if abs(100 - mape_accuracy * 100) > 100:
            nn_accuracy = r2_accuracy * 100; self.r2 = True
        else:
            nn_accuracy = 100 - mape_accuracy * 100

        self.forecast.append(forecast)
        self.nn_accuracy.append(nn_accuracy)
        self.total_data_explored.append(total_data_explored)


    def get_best_model_values(self):
        max_realistic = [x for x in self.nn_accuracy if abs(x) <=100]
        max_value = max(max_realistic, key=abs) if max_realistic else max(l for l in self.nn_accuracy)
        max_index = self.nn_accuracy.index(max_value)
        metric_measure = "R-Squared Variance" if self.r2 else "Mean Absolute Accuracy"
        print("THE TIMESERIES SELECTED MODEL IS:")
        print(self.model_name[max_index])
        print(max_realistic)
        print(self.nn_accuracy)
        return self.forecast[max_index], abs(max_value), metric_measure, self.total_data_explored[max_index], self.model_name[max_index]



class GroupWithCommandOptions(click.Group):
    """Allow application of options to group with multi command"""

    def add_command(self, cmd, name=None):
        click.Group.add_command(self, cmd, name=name)

        # add the group parameters to the command
        for param in self.params:
            cmd.params.append(param)

        # hook the commands invoke with our own
        cmd.invoke = self.build_command_invoke(cmd.invoke)
        self.invoke_without_command = True

    def build_command_invoke(self, original_invoke):
        def command_invoke(ctx):
            """insert invocation of group function"""

            # separate the group parameters
            ctx.obj = dict(_params=dict())
            for param in self.params:
                name = param.name
                ctx.obj["_params"][name] = ctx.params[name]
                del ctx.params[name]

            # call the group function with its parameters
            params = ctx.params
            ctx.params = ctx.obj["_params"]
            self.invoke(ctx)
            ctx.params = params

            # now call the original invoke (the command)
            original_invoke(ctx)

        return command_invoke