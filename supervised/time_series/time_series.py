import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
#from mpl_finance import candlestick_ohlc

class TimeSeries:
    @staticmethod
    def display_bollinger_bands(ts_df, col_name, window_size, figsize=(16,9), **plot_option):
        df_plot = ts_df[[col_name]].copy()
        df_plot['rolling_mean'] = df_plot[col_name].rolling(window=window_size).mean()
        df_plot['upper_band'] = df_plot['rolling_mean'] + 2 * df_plot['rolling_mean'].rolling(window=window_size).std()
        df_plot['lower_band'] = df_plot['rolling_mean'] - 2 * df_plot['rolling_mean'].rolling(window=window_size).std()
        plot_option['figsize'] = figsize
        df_plot.plot(**plot_option)
        plt.show()

    @staticmethod
    def display_pair_plot(ts_df, col_names, figsize=(16,9), **plot_option):
        df_plot = ts_df[col_names].copy()
        sns.pairplot(df_plot.dropna())
        plot_option['fig_size'] = figsize
        plt.show(**plot_option)

    # @staticmethod
    # def display_candle_stick():
    #     df_reset = df.loc['2017-02'].reset_index()
    #     fig, ax = plt.subplots()
    #     df_reset['date_time'] = df_reset['index'].apply(lambda x:mdates.date2num(x))
    #     plot_data = [tuple(val) for val in df_reset[['date_time', 'mean_20', 'upper', 'lower', 0]].values]
    #     candlestick_ohlc(ax, plot_data, width=0.4)
    #     plt.show()

    @staticmethod
    def plot_stability_summary(ts_df, col_name, leg):
        df = ts_df[[col_name]].copy()
        df['var'] = df[col_name].pct_change(leg).var()
        df['var'].plot()
        plt.show()
        return df

date_rng_2017 = pd.date_range('1/1/2017', periods=365, freq='D')
df = pd.DataFrame(np.random.randn(365).cumsum(), index=date_rng_2017)
TimeSeries.display_bollinger_bands(df, 0,20)