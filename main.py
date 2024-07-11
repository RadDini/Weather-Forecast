import datetime
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Function(Enum):
    INFO = 1
    TEMP_MAX_HISTPLOT = 2
    TEMP_MAX_FACEGRID = 3
    PRECIP_HISTPLOT = 4
    WEATHER_COUNTPLOT = 5
    WEATHER_PIECHART = 6
    LR_DEFAULT = 7
    LR_RANDOM = 8
    SVR_DEFAULT = 9


def dataframe_add_months_years(dataframe: pd.DataFrame) -> pd.DataFrame:
    # adds month and year columns to the dataframe.
    # data frame should have a column date of type datetime

    return dataframe.assign(month=dataframe['date'].dt.month, year=dataframe['date'].dt.year)


def dataset_info(dataframe: pd.DataFrame):
    # prints information about the dataset

    print(dataframe)

    dataframe.info(verbose=True)

    print("Null values:")
    print(str(dataframe.isnull().sum()))

    print(f"Duplicate values: {dataframe.duplicated().sum()}")

    print(f"Min temp = {dataframe.min()['temp_min']}")

    print(f"Max temp = {dataframe.max()['temp_max']}")

    print(f"Most common weather: ")
    print(dataframe.drop('date', axis=1).mode())


def temp_max_histplot(dataframe: pd.DataFrame):
    seaborn.histplot(dataframe, x='temp_max')
    plt.title("Temperature distribution")
    plt.show()


def rotate_and_skip_ticks(g):
    for ax in g.axes.flat:
        labels = ax.get_xticklabels()  # get x labels
        for i, l in enumerate(labels):
            if i % 2 == 0:
                labels[i] = ''  # skip even labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30)  # set new labels


def temp_max_facegrid_lineplot(dataframe: pd.DataFrame):
    # prints 4 line plots for the maximum temperature
    dataframe['month_str'] = dataframe['date'].dt.strftime("%b")  # set month to its abbreviated name

    g = seaborn.FacetGrid(dataframe, col='year', col_wrap=4)
    g.map(seaborn.lineplot, 'month_str', 'temp_max')

    g.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    g.fig.suptitle('Maximum temperature distribution by year')

    rotate_and_skip_ticks(g)

    plt.show()


def precipitation_facegrid_scatterplot(dataframe: pd.DataFrame):
    # prints 4 scatter plots for precipitations
    dataframe['month_str'] = dataframe['date'].dt.strftime("%b")  # set month to its abbreviated name

    plt.title("Precipitation distribution")

    g = seaborn.FacetGrid(dataframe, col='year', col_wrap=4)
    g.map(seaborn.scatterplot, 'month_str', 'precipitation', )

    g.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    g.fig.suptitle('Precipitation distribution by year')

    rotate_and_skip_ticks(g)

    plt.show()


def weather_countplot(dataframe: pd.DataFrame):
    # prints a bar plot that shows how many of each weather types have been recorded

    seaborn.countplot(dataframe, x='weather')
    plt.title("Weather distribution")
    plt.show()


def weather_piechart(dataframe: pd.DataFrame):
    # prints the distribution of weather types as a pie chart

    weather_type_counts = dataframe['weather'].value_counts()
    plt.pie(weather_type_counts, labels=weather_type_counts.keys(), autopct='%1.1f%%')
    plt.title("Weather piechart")
    plt.show()


def lr_predictor_with_split(dataframe: pd.DataFrame, test_size=0.2, no_plot=False):
    # uses linear regression to predict the maximum temperature for a given split
    # prints a line plot showing the actual data vs the predicted data

    temps_max = dataframe['temp_max']

    features = dataframe[['precipitation', 'month', 'year', 'wind', 'date', 'temp_min']].dropna()

    X_train, X_test, y_train, y_test = train_test_split(features, temps_max, test_size=test_size, random_state=42,
                                                        shuffle=False)

    reg = LinearRegression().fit(X_train.drop('date', axis=1), y_train)
    print(pd.DataFrame(reg.coef_, features.drop('date', axis=1).columns, columns=['Coeff']))

    predictions = reg.predict(X_test.drop('date', axis=1))
    print("MSE:", metrics.mean_squared_error(y_test, predictions))

    if no_plot:
        return

    test_size = int(test_size * len(dataframe.index))
    train_size = len(dataframe.index) - test_size
    random_split_size = int(0.11 * len(dataframe.index))

    random_split_first_date = dataframe.iloc[test_size - random_split_size + train_size]['date']
    print(random_split_first_date)


    y_predicted = np.array(predictions)
    y_actual = np.array(y_test)
    date = np.array(X_test['date'])

    plt.plot(date, y_predicted)
    plt.plot(date, y_actual)

    orange_patch = mpatches.Patch(color='orange', label='Actual data')
    blue_patch = mpatches.Patch(color='blue', label='Predicted data')

    if test_size > random_split_size:
        plt.axvline(random_split_first_date, color='r')
        red_patch = mpatches.Patch(color='red', label='Data intersection with smaller test size')
        plt.legend(handles=[red_patch, orange_patch, blue_patch])
    else:
        plt.legend(handles=[orange_patch, blue_patch])

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.AutoDateFormatter(locator)

    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title(f"Linear Regression with test size {int(test_size)}")
    plt.xlabel("Date")
    plt.ylabel("Maximum temperature")

    plt.show()


def lr_predictor_random_split(dataframe: pd.DataFrame):
    lr_predictor_with_split(dataframe, test_size=0.11)


def lr_predictor_default_split(dataframe: pd.DataFrame):
    lr_predictor_with_split(dataframe)


def svr_predictor_default_split(dataframe: pd.DataFrame):
    # uses LinearSVR to predict the minimum temperature for a given split
    # prints a line plot showing the actual data vs the predicted data

    reg = make_pipeline(StandardScaler(),
                        LinearSVR(random_state=0, tol=1e-5))

    temps_min = dataframe['temp_min']

    features = dataframe[['precipitation', 'month', 'year', 'wind', 'date', 'temp_max']].dropna()

    X_train, X_test, y_train, y_test = train_test_split(features, temps_min, test_size=0.2, random_state=42,
                                                        shuffle=False)

    reg.fit(X_train.drop('date', axis=1), y_train)
    print("Score: ", reg.score(X_test.drop('date', axis=1), y_test))

    predictions = reg.predict(X_test.drop('date', axis=1))
    print("MSE:", metrics.mean_squared_error(y_test, predictions))

    y_predicted = np.array(predictions)
    y_actual = np.array(y_test)
    date = np.array(X_test['date'])

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.AutoDateFormatter(locator)

    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.plot(date, y_predicted)
    plt.plot(date, y_actual)
    plt.xlabel("Date")
    plt.ylabel("Minimum temperature")
    plt.title("LinearSVR for minimum temperature")

    plt.show()


def main():
    df = pd.read_csv('seattle-weather.csv')

    # turn date into datetime
    df['date'] = pd.to_datetime(df['date'])
    df = dataframe_add_months_years(df)

    # implemented fucntionalities of the program
    print("Functions: ")
    for function in Function:
        print(function.name, ":", function.value)

    while True:
        choice = Function(int(input("Choose function: ")))
        print(choice)

        match choice:
            case Function.INFO:
                dataset_info(df)

            case Function.TEMP_MAX_HISTPLOT:
                temp_max_histplot(dataframe=df)

            case Function.TEMP_MAX_FACEGRID:
                temp_max_facegrid_lineplot(dataframe=df)

            case Function.PRECIP_HISTPLOT:
                precipitation_facegrid_scatterplot(dataframe=df)

            case Function.WEATHER_COUNTPLOT:
                weather_countplot(dataframe=df)

            case Function.WEATHER_PIECHART:
                weather_piechart(dataframe=df)

            case Function.LR_DEFAULT:
                lr_predictor_default_split(dataframe=df)

            case Function.LR_RANDOM:
                lr_predictor_random_split(dataframe=df)

            case Function.SVR_DEFAULT:
                svr_predictor_default_split(dataframe=df)


if __name__ == '__main__':
    main()
