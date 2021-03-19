import requests
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


r = requests.get('https://nepalcorona.info/api/v1/data/nepal').json()
df=pd.DataFrame.from_dict(r)
#cleaning
df.reset_index(drop=True,inplace=True)
df.drop(['source','updated_at','latest_sit_report','pending_result'],inplace=True,axis=1)
df.drop([0,2,3,4,5,6,7],axis=0,inplace=True)

# formatting
columnsTitles=["tested_total","tested_positive","tested_negative","recovered","deaths","in_isolation","quarantined"]
df=df.reindex(columns=columnsTitles)
print(df)

def piechart():
        # ------------------------------------------------------DRAWING PIE CHART-----------------------------------------------                                            
        # creating size list for pieplot from dataframe elements
        isolation=df['in_isolation'].tolist()
        recov=df['recovered'].tolist()
        dec=df['deaths'].tolist()  #deceased
        sizes=[isolation[0],recov[0],dec[0]]   #isolation,recov and deaths all are singlelist,taking their value into new sizes list

        labels = ['Active','Recovered','Deceased']
        color= ['yellow','green','red']
        explode = [0,0.01,0.0]

       
    
        plt.figure(figsize= (6,6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode,colors = color)
        centre_circle = plt.Circle((0,0),0.70,fc='white')

        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title('Nepal Covid Cases',fontsize = 20)
        plt.axis('equal')  
        plt.tight_layout()
        plt.show()

piechart()