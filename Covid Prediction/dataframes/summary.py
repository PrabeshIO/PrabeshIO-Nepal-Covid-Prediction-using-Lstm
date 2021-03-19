import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq
import matplotlib.pyplot as plt
import matplotlib

# total summmary 
# start
def summmary():
        r = requests.get('https://nepalcorona.info/api/v1/data/nepal').json()
        df=pd.DataFrame.from_dict(r)

        df.reset_index(drop=True,inplace=True)
        df.drop(['source','updated_at','latest_sit_report','pending_result'],inplace=True,axis=1)

        df.drop([0,2,3,4,5,6,7],axis=0,inplace=True)
        columnsTitles=["tested_total","tested_positive","tested_negative","recovered","deaths","in_isolation","quarantined"]
        df=df.reindex(columns=columnsTitles)
        return(df)

        





