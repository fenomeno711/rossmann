from sklearn.preprocessing import StandardScaler

from datetime import datetime

# Standard Scaler
def std_scaler(df):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df_scaled_pd = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled_pd


def convert2day(date_string):
    return datetime.strptime(date_string,'%Y-%m-%d').day


def convert2month(date_string):
    return datetime.strptime(date_string,'%Y-%m-%d').month


def convert2year(date_string):
    return datetime.strptime(date_string,'%Y-%m-%d').year


def convert2timestamp(date_string):
    return datetime.strptime(date_string,'%Y-%m-%d').timestamp()


