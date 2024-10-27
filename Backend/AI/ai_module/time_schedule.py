import numpy as np
from datetime import datetime, timedelta

class AutoregressionModel:
    def __init__(self, train_size=1, min_corr=0, num_lags=-1):
        self.train_size = train_size
        self.min_corr = min_corr
        self.num_lags = num_lags
        self.datetime_series = []
        self.datetime_intervals = []
        self.significant_lags = []
    
    def fit(self, datetime_series):
        self.datetime_series = datetime_series
        self.datetime_intervals = self.__generate_time_intervals(datetime_series)
        if self.num_lags == -1:
            self.num_lags = len(self.datetime_intervals) - 1
        if self.num_lags > 0:
            lags = range(1, self.num_lags)
            autocorrelations = np.nan_to_num([self.__autocorrelation(self.datetime_intervals, lag) for lag in lags])
            self.significant_lags = sorted(zip(lags, autocorrelations), key=lambda x: x[1], reverse=True)[:self.train_size]
        return self

    def predict_next_value(self):
        if self.num_lags == 0:
            return self.datetime_series[-1] + timedelta(seconds=self.datetime_intervals[-1])
        predicted = 0
        for lag, corr_coef in self.significant_lags:
            predicted += self.datetime_intervals[-lag] * corr_coef
        predicted /= sum([abs(x) for _, x in self.significant_lags])
        return self.datetime_series[-1] + timedelta(seconds=self.datetime_intervals[-1] if predicted < 0 else predicted)
    
    @staticmethod
    def __generate_time_intervals(series):
        return [(series[i+1]-series[i]).total_seconds() for i in range(len(series)-1)]
    
    # Определение коэффициентов автокорреляции
    @staticmethod
    def __autocorrelation(series, lag):
        y1 = series[:len(series) - lag]
        y2 = series[lag:]
        return np.corrcoef(y1, y2)[0, 1]

# Пример использования
if __name__ == "__main__":
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    datetime_series = [
        datetime.strptime("2024-10-11 19:41:03.709297", date_format),
        datetime.strptime("2024-10-13 20:01:02.709297", date_format),  
        datetime.strptime("2024-10-16 23:41:03.101", date_format),
        datetime.strptime("2024-10-20 20:11:17.709297", date_format),
        datetime.strptime("2024-10-22 19:55:56.709297", date_format),  
        datetime.strptime("2024-10-25 19:41:03.709297", date_format),
    ]

    res = AutoregressionModel().fit(datetime_series).predict_next_value()
    print(res)
