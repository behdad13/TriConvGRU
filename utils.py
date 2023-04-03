from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
from preprocessing import descale
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def calculate_metrics(values_descaled, prediction_descaled):
    result_metrics = {
                      'MAE' : mean_absolute_error(values_descaled, prediction_descaled),
                      'MSE' : mean_squared_error(values_descaled, prediction_descaled),
                      'R2'  : r2_score(values_descaled, prediction_descaled),
                      'MAPE': mean_absolute_percentage_error(values_descaled, prediction_descaled)
                      }
    print("Root Mean Squared Error :  ", result_metrics["MSE"]**0.5)
    print("R^2                :       ", result_metrics["R2"])
    print("Mean Absolute Error:       ", result_metrics["MAE"])
    print("MAPE               :       ", result_metrics["MAPE"])
    
    return result_metrics


def create_forecast_table(preds, true):
    # Create an empty DataFrame with the desired columns
    df = pd.DataFrame(columns=['forcasted_price_1h', 'forcasted_price_2h', 'forcasted_price_3h',
                               'real_price_1h', 'real_price_2h', 'real_price_3h'])

    # Fill the DataFrame using the loop
    for i in range(len(preds)):
        prediction_descaled = descale(preds[[i]].reshape(-1))
        values_descaled = descale(true[[i]].reshape(-1))
        new_row = {'forcasted_price_1h': prediction_descaled[0],
                   'forcasted_price_2h': prediction_descaled[1],
                   'forcasted_price_3h': prediction_descaled[2],
                   'real_price_1h': values_descaled[0],
                   'real_price_2h': values_descaled[1],
                   'real_price_3h': values_descaled[2]}
        df = df.append(new_row, ignore_index=True)
    return df


def plot_forecast(df):
    print('Error by our model for next 1 hour prediction:')
    error_1h = calculate_metrics(df.forcasted_price_1h, df.real_price_1h)
    print("\n")

    print('Error by our model for next 2 hour prediction:')
    error_2h = calculate_metrics(df.forcasted_price_2h, df.real_price_2h)
    print("\n")

    print('Error by our model for next 3 hour prediction:')
    error_3h = calculate_metrics(df.forcasted_price_3h, df.real_price_3h)
    print("\n")

    figure(figsize=(19, 6), dpi=250)
    dates = range(len(df.real_price_1h))
    plt.plot(dates[0:200], df.forcasted_price_1h[0:200], "--", label="prediction")
    plt.plot(dates[0:200], df.real_price_1h[0:200], "-", label="real")
    plt.xticks(rotation=0)
    plt.legend()
    plt.savefig('result/forecast_1h.png', dpi=100)
    plt.close()

    figure(figsize=(19, 6), dpi=250)
    dates = range(len(df.real_price_2h))
    plt.plot(dates[0:200], df.forcasted_price_2h[0:200], "--", label="prediction")
    plt.plot(dates[0:200], df.real_price_2h[0:200], "-", label="real")
    plt.xticks(rotation=0)
    plt.legend()
    plt.savefig('result/forecast_2h.png', dpi=100)
    plt.close()

    figure(figsize=(19, 6), dpi=250)
    dates = range(len(df.real_price_3h))
    plt.plot(dates[0:200], df.forcasted_price_3h[0:200], "--", label="prediction")
    plt.plot(dates[0:200], df.real_price_3h[0:200], "-", label="real")
    plt.xticks(rotation=0)
    plt.legend()
    plt.savefig('result/forecast_3h.png', dpi=100)
    plt.close()

    

