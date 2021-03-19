from bs4 import BeautifulSoup as soup
import pandas as pd
# import numpy as np
from urllib.request import urlopen as uReq
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

# df=pd.read_csv('owid-covid-data.csv')
# print df
def get_timeline():

    df=pd.read_csv("dataframes\owid-covid-data.csv")
    last_date= df['date'].tolist()
    x= len(df['date'])
    return (df, last_date[x-1])



# print(get_timeline())