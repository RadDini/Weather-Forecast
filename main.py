import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import datetime


def dataset_info(dataframe: pd.DataFrame):
    """ TODO:
    """

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
    """ TODO:
    """

    seaborn.histplot(dataframe, x='temp_max')
    plt.title("Temperature distribution")
    plt.show()


def temp_max_facegrid_lineplot(dataframe: pd.DataFrame):
    """ TODO:
    """
    months = []
    years = []
    for date in dataframe['date']:
        dt = datetime.datetime.strptime(date, '%Y-%m-%d')
        months.append(int(dt.month))
        years.append(int(dt.year))

    dataframe = dataframe.assign(month=months, year=years)

    g = seaborn.FacetGrid(dataframe, col='year', col_wrap=4)
    g.map(seaborn.lineplot, 'month', 'temp_max')

    plt.show()


def precipitation_facegrid_scatterplot(dataframe: pd.DataFrame):
    """ TODO:
    """


def weather_countplot(dataframe: pd.DataFrame):
    """ TODO:
    """


def weather_piechart(dataframe: pd.DataFrame):
    """ TODO:
    """


def lr_predictor_random_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def lr_predictor_default_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def svr_predictor_default_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def main():
    df = pd.read_csv('seattle-weather.csv')
    # dataset_info(df)

    # temp_max_histplot(dataframe=df)

    temp_max_facegrid_lineplot(dataframe=df)


if __name__ == '__main__':
    main()
