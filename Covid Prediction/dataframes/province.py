from bs4 import BeautifulSoup as soup
import pandas as pd
import numpy as np
from urllib.request import urlopen as uReq


my_url = "https://en.wikipedia.org/wiki/Template:COVID-19_pandemic_data/Nepal_medical_cases_by_province_and_district"
uClient= uReq(my_url)
page_html= uClient.read()

uClient.close()
page_soup= soup(page_html,"html.parser")

table=page_soup.findAll('table',{'class':"wikitable"})      #selecting table
head=table[0].findAll('tr',{'class':'covid-sticky'})        #selecting fisr row or heading with class covid-sticky

columns=[] #columns heading

for i in [0,2,3]:
    
         col=head[0].findAll('th')[i].text
         columns.append(col.replace("\n",""))
            
# print(columns)

# 2-pr 1
# 17- pr 2
# 26-bagmati
# 40-gandaki
# 52- pr 5
# 65- karnali 
# 76-sudurpaschim

# 86- whole nepal

def replace_symbol(x):
    
    x=x.replace("\n","")
    x=x.replace(",","")
    return x


    
def get_province():
        body=table[0].findAll('tr')             #all rows 
        cases=[]                                #for making list to  make dataframe
        
        for i in [2,17,26,40,52,65,76]:
            
                cell=body[i].findAll('td')       # where i is the province index
                data=[]                          #for storing each row
            
                for j in [0,2,3]:                # 0- province name ,2-cases  and 3- deaths
                    
                    x=(cell[j]).text
                    x=replace_symbol(x)
                    
                    if j!=0:
                        x= int(x)
                    data.append(x)
                cases.append(data)     
        
        df=pd.DataFrame(cases, columns=columns)
        df.index+=1
        return df

# x=get_province()
# print(x)





