import pandas as pd


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


def temp_max_facegrid_lineplot(dataframe: pd.DataFrame):
    """ TODO:
    """


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
    dataset_info(df)


if __name__ == '__main__':
    main()
