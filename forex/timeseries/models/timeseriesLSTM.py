import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import SGD, Adam, Adadelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping, LearningRateScheduler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from .helpers import read_pickle
from sklearn.utils import shuffle
import os
from django.conf import settings
from keras import backend as K
import statsmodels.api as sm



class Lstm_univarient(object):
    def __init__(self,**kwargs):
        super().__init__()

        self.model=False

        if 'activation_functon' in kwargs:
            self.activation_function = kwargs['activation_function']
        else:
            self.activation_function = 'relu'

        if 'model_optimizer' in kwargs:
            self.model_optimizer = kwargs['model_optimizer']
        else:
            self.model_optimizer = 'Adam'

        if 'model_loss' in kwargs:
            self.model_loss = kwargs['model_loss']
        else:
            self.model_loss = 'mean_squared_error'

        if 'model_metrics' in kwargs:
            self.model_metrics = kwargs['model_metrics']
        else:
            self.model_metrics = ['mean_absolute_error']

        if 'kernal_initializer' in kwargs:
            self.kernal_init = kwargs['kernal_initializer']
        else:
            self.kernal_init = 'normal'

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = 0.01

        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 32

        if 'epochs_number' in kwargs:
            self.epochs_number = kwargs['epochs_number']
        else:
            self.epochs_number = 30

        if 'n_neuron' in kwargs:
            self.n_neuron = kwargs['n_neuron']
        else:
            self.n_neuron = 100

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = 0

        if 'dropout' in kwargs:
            self.dropout = kwargs['dropout']
        else:
            self.dropout = 0.25

        if 'look_back' in kwargs:
            self.look_back = kwargs['look_back']
        else:
            self.look_back = 7

        self.decay = self.learning_rate / self.epochs_number
        self.__GPU_check()

    def __GPU_check(self):
        ls = tf.config.list_physical_devices('GPU')
        if len(ls) > 0:
            if self.verbose == 1:
                print("[INFO] Oh Yes.. You are working with GPU support !")
            tf.config.experimental.set_memory_growth(ls[0], True)
        else:
            if self.verbose == 1:
                print("[INFO] Boo.. No GPU Support !!")

    def _getBalancedData(self, x,y):
        oversample = SMOTE(sampling_strategy='minority', random_state=7)
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=7)
        X_smote, Y_smote = oversample.fit_resample(x, y)
        X, Y = undersample.fit_resample(X_smote, Y_smote)
        return X,Y

    def reading_csv(self, row_start=0, row_stop=500, col_start=0, col_stop=-1):
        # dataframe = read_csv(os.path.join(self.path, self.filename))
        dataframe = read_pickle(self.path)
        row_stop = len(dataframe)
        # DeepLearning.COLUMN_NAMES = dataframe.columns
        dataset = dataframe.values
        # split into input (X) and output (Y) variables
        X = dataset[row_start:row_stop, col_start:col_stop].astype(float)
        self.numberof_input_fields = X.shape[1]
        if col_stop < -1:
            Y_raw = dataset[row_start:row_stop, col_stop:]
        else:
            Y_raw = dataset[row_start:row_stop, col_stop]
        # encode class values as integers
        Y = self._preprocessing_encoding(Y_raw)
        # x_bal, y_bal = shuffle(self._getBalancedData(X, Y))
        x_bal,y_bal = shuffle(X,Y)
        return x_bal, y_bal

    def __prepare_dataset(self,dataset,split):
        dataset = np.reshape(dataset, (-1, 1))

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        transformed_data = self.scaler.fit_transform(dataset)

        train_size = int(len(transformed_data) * split)
        test_size = len(transformed_data) - train_size
        if self.verbose == 1:
            print('[INFO] Train data {} ,test data {}'.format(train_size,test_size))
        # return train, test
        return transformed_data[0:train_size, :], transformed_data[train_size:len(dataset), :],transformed_data

    # convert an array of values into a dataset matrix
    def __create_dataset(self,dataset, look_back):
        X, Y = [], []
        if self.verbose == 1: print('[INFO] Creating Datasets ..')
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    # reshape input to be [samples, time steps, features]
    def __reshape_data(self,array_input,time_steps):
        return np.reshape(array_input, (array_input.shape[0],time_steps,array_input.shape[1]))

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true[0]), np.array(y_pred[:,0])
        dinom = y_true
        # changing denominator zero to 1
        # using this to change only denominator to skip divide by zero error
        # eventually zero values will result to be zero
        dinom[dinom == 0] = dinom.mean()
        # dinom[dinom < 0] = abs(dinom[[dinom < 0]])
        result = np.abs((y_true - y_pred) / (dinom)).mean()
        return result

    @staticmethod
    def mean_squared_error(X,Y):
        return mean_absolute_error(X[0], Y[:, 0])

    def RMS(self,X,Y):
        return np.sqrt(self.mean_absolute_percentage_error(X, Y))

    @staticmethod
    def weighted_mean_absolute_error(true, pred):
        true, pred = np.array(true[0]), np.array(pred[:,0])

        return abs(true - pred).sum() / abs(true).sum()

    @staticmethod
    def r_squared_adj(true, pred):
        true, pred = np.array(true[0]), np.array(pred[:, 0])
        true_addC = sm.add_constant(true)

        return (sm.OLS(pred, true_addC).fit()).rsquared_adj

    def __model(self,X_train):
        model = Sequential()
        model.add(LSTM(self.n_neuron, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(self.n_neuron//2, activation=self.activation_function))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))

        if self.model_optimizer == 'SGD':
            optmi = SGD(learning_rate=self.learning_rate,
                        decay = self.decay,
                        momentum= 0.8)
        elif self.model_optimizer == 'Adam':
            optmi = Adam(learning_rate=self.learning_rate)
        elif self.model_optimizer == 'Adadelta':
            optmi = Adadelta(learning_rate=self.learning_rate)
        else:
            print("[INFO] Unknown Optimizer (Using Deafault SGD optimizer)")
            optmi = SGD(learning_rate=self.learning_rate,
                        decay=self.decay,
                        momentum=0.8)

        model.compile(loss=self.model_loss,
                      optimizer=optmi,
                      metrics=[self.model_metrics])

        self.model = model
        return model

    # define the learning rate change
    def __exp_decay(self,epoch):
        lrate = self.learning_rate * np.exp(-self.decay * epoch)
        return float(lrate)

    def __fit(self,x_train,y_train,x_test=None,y_test=None):

        lr_rate = LearningRateScheduler(self.__exp_decay, verbose=self.verbose)
        callbacks = EarlyStopping(monitor='val_loss', patience=10)
        callbacks_list = [callbacks, lr_rate]

        model = self.__model(x_train)

        if self.verbose == 1:
            print('[INFO]')
            print(model.summary())
            print('[INFO] Fitting the model ..')

        history = model.fit(x_train, y_train,
                            epochs=self.epochs_number,
                            batch_size=self.batch_size,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks_list,
                            verbose=self.verbose, shuffle=False)

        return history

    def prepare_fit(self,dataset,split=0.2,target=None,time_period=1):
        """
        Method to prepare the dataset and fit the model defined.
        Args:
            dataset: Dataset with index as datetime
            split: split ratio of test data
            target: targeted column to forecast
            time_period: time period interval between dates

        Returns: keras.callbacks.History

        """
        if type(dataset).__name__ != 'DataFrame':
            raise ValueError('Dataframe is require not {}'.format(type(dataset)))
        if target == None:
            print('[INFO] Using first column, {} as target'.format(dataset.columns[0]))
            target = dataset.columns[0]

        self.time_index = dataset.index
        if dataset.shape[1] != 1:
            dataset = pd.DataFrame(dataset[target]).set_axis(self.time_index)

        train,test,total_data = self.__prepare_dataset(dataset,1-split)

        if len(test) < (self.look_back + 1):
            print('[ERROR] Unsufficient data to design a dataset\n Possible fixes:\n [1] Decrease split ratio, '
                  '[2] Increase data size')
            raise ValueError('Unsufficient data to design a dataset')

        X_train , Y_train = self.__create_dataset(train,self.look_back)
        X_test, Y_test = self.__create_dataset(test,self.look_back)
        X_total_data, Y_total_data = self.__create_dataset(total_data,self.look_back)

        X_train = self.__reshape_data(X_train,time_period)
        X_test = self.__reshape_data(X_test,time_period)
        X_total_data = self.__reshape_data(X_total_data,time_period)

        self.dataset = dataset
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_total_data = X_total_data
        self.Y_total_data = Y_total_data

        return self.__fit(X_train,Y_train,X_test,Y_test)

    # def actual_vs_predicted(self):
    #     predict = self.scaler.inverse_transform(self.__predict(self.X_total_data))
    #     actual = self.scaler.inverse_transform([self.Y_total_data])
    #     return actual,predict

    def accuracy(self):
        # make predictions
        test_predict = self.scaler.inverse_transform(self.__predict(self.X_test))
        # test_predict = self.__predict(self.X_test)
        # invert predictions
        y_test = self.scaler.inverse_transform([self.Y_test])

        # print('Mean Absolute Error:', self.mean_squared_error(y_test, test_predict))
        # print('Root Mean Squared Error:', self.RMS(y_test, test_predict))

        return {'Mean_Absolute_Error':self.mean_absolute_percentage_error(y_test, test_predict),
                'Root_Mean_Error':self.RMS(y_test, test_predict),
                'WMAE':self.weighted_mean_absolute_error(y_test,test_predict),
                'R2': self.r_squared_adj(y_test, test_predict)

                }

    def __predict(self,X):
        return self.model.predict(X)


    def predict(self,X):
        predict = self.__predict(X)
        return self.scaler.inverse_transform(predict)

    def __find_time_period(self):
        difference = self.time_index[-1] - self.time_index[-2]
        if difference.seconds == (60): time = "MIN"
        elif difference.seconds == (60 * 60): time = "HRS"
        elif difference.seconds == (60 * 60 * 24): time = "DAY"
        elif difference.seconds == (60 * 60 * 24 * 30): time = "MON"
        else: time = "YER"

        return time

    def __forecast_time_list(self,_range):
        difference = self.time_index[-1] - self.time_index[-2]
        final_time = self.time_index[-1]
        list_of_time =[]

        for _ in range(_range):
            final_time = final_time + difference
            list_of_time.append(final_time)

        return list_of_time

    def forecast(self,data_points=50):
        """
        This method forecast next (data_points) and return dataframe
        Args:
            data_points: number of forecasting from last date

        Returns: dataframe with datetime as index and 'Prediction' column

        """
        final_prediction = []

        first_to_predict = np.append(self.X_test[-1], self.Y_test[-1])[1:].reshape(1, 1, self.X_test.shape[2])
        predicted = self.__predict(first_to_predict)
        final_prediction.append(self.predict(first_to_predict)[0][0])

        to_predict = first_to_predict

        for _ in range(data_points-1):
            to_predict = np.append(to_predict[-1], predicted[-1])[1:].reshape(1, 1, to_predict.shape[2])
            predicted = self.__predict(to_predict)
            final_prediction.append(self.predict(to_predict)[0][0])

        return pd.DataFrame(final_prediction, columns=['Predicted'],index=self.__forecast_time_list(data_points))

    def actual_vs_predicted(self,data_points=50):
        final_prediction = []

        first_to_predict = np.append(self.X_test[-data_points//2], self.Y_test[-data_points//2])[1:].reshape(1, 1, self.X_test.shape[2])
        predicted = self.__predict(first_to_predict)
        final_prediction.append(self.predict(first_to_predict)[0][0])

        to_predict = first_to_predict

        for _ in range(data_points//2 - 1):
            to_predict = np.append(to_predict[-1], predicted[-1])[1:].reshape(1, 1, to_predict.shape[2])
            predicted = self.__predict(to_predict)
            final_prediction.append(self.predict(to_predict)[0][0])

        return pd.DataFrame(final_prediction, columns=['past_predicted'], index=self.__forecast_time_list(data_points//2))

    def serialize_model(self, model, model_name='deep_learning_timeseries.h5',
                        model_repo='models_vault', *args, **kwargs):
        user_directory = str(self.user_directory)
        serialize_sub_directory_path = '/media/models_vault/deep_learning_timeseries/'
        model_repo = os.path.join(settings.BASE_DIR + serialize_sub_directory_path, user_directory)
        if not os.path.exists(model_repo):
            os.makedirs(model_repo, exist_ok=True)
        self.model_repo = model_repo
        self.final_model_path = os.path.join(model_repo, model_name)
        model.save(self.final_model_path)
        K.clear_session()
        return (self.final_model_path)