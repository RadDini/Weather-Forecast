import datetime
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Function(Enum):
    INFO = 1
    TEMP_MAX_HISTPLOT = 2
    TEMP_MAX_FACEGRID = 3
    PRECIP_HISTPLOT = 4
    WEATHER_COUNTPLOT = 5
    WEATHER_PIECHART = 6
    LR_DEFAULT = 7
    LR_RANDOM = 8


def dataframe_add_months_years(dataframe: pd.DataFrame) -> pd.DataFrame:
    months = []
    years = []
    for date in dataframe['date']:
        dt = datetime.datetime.strptime(date, '%Y-%m-%d')
        months.append(int(dt.month))
        years.append(int(dt.year))

    return dataframe.assign(month=months, year=years)


def dataset_info(dataframe: pd.DataFrame):
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


def temp_max_facegrid_lineplot(dataframe: pd.DataFrame):
    dataframe = dataframe_add_months_years(dataframe)

    g = seaborn.FacetGrid(dataframe, col='year', col_wrap=4)
    g.map(seaborn.lineplot, 'month', 'temp_max')

    plt.show()


def precipitation_facegrid_scatterplot(dataframe: pd.DataFrame):
    dataframe = dataframe_add_months_years(dataframe)

    g = seaborn.FacetGrid(dataframe, col='year', col_wrap=4)
    g.map(seaborn.scatterplot, 'month', 'precipitation')

    plt.show()


def weather_countplot(dataframe: pd.DataFrame):
    seaborn.countplot(dataframe, x='weather')
    plt.show()


def weather_piechart(dataframe: pd.DataFrame):
    weather_type_counts = dataframe['weather'].value_counts()
    plt.pie(weather_type_counts, labels=weather_type_counts.keys(), autopct='%1.1f%%')
    plt.title("Weather piechart")
    plt.show()


def lr_predictor_with_split(dataframe: pd.DataFrame, test_size=0.2):
    dataframe = dataframe_add_months_years(dataframe)
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    temps_max = dataframe['temp_max']

    features = dataframe[['precipitation', 'month', 'year', 'wind', 'date', 'temp_min']].dropna()

    X_train, X_test, y_train, y_test = train_test_split(features, temps_max, test_size=test_size, random_state=42,
                                                        shuffle=False)

    reg = LinearRegression().fit(X_train.drop('date', axis=1), y_train)
    print(pd.DataFrame(reg.coef_, features.drop('date', axis=1).columns, columns=['Coeff']))

    predictions = reg.predict(X_test.drop('date', axis=1))
    print("MSE:", metrics.mean_squared_error(y_test, predictions))

    y_predicted = np.array(predictions)
    y_actual = np.array(y_test)
    date = np.array(X_test['date'])

    plt.plot(date, y_predicted)
    plt.plot(date, y_actual)

    plt.show()


def lr_predictor_random_split(dataframe: pd.DataFrame):
    lr_predictor_with_split(dataframe, test_size=0.11)


def lr_predictor_default_split(dataframe: pd.DataFrame):
    lr_predictor_with_split(dataframe)


def svr_predictor_default_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def main():
    df = pd.read_csv('seattle-weather.csv')

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


if __name__ == '__main__':
    main()
