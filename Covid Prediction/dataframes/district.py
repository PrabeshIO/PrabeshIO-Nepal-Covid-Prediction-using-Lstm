from bs4 import BeautifulSoup as soup
import pandas as pd
import numpy as np
from urllib.request import urlopen as uReq


my_url = 'https://kathmandupost.com/covid19'
uClient= uReq(my_url)
page_html= uClient.read()

uClient.close()
page_soup= soup(page_html,"html.parser")
# print(page_soup)

rows=  page_soup.findAll('tr',{'class':'district-row'})
# print(row)
districts=[]
cases=[]
death=[]
for i in range(77):
    row= rows[i].findAll('td')
    districts.append(row[0].text)  #district names
    cases.append(row[1].text)    
    death.append(row[2].text)
# print(districts)
# print(cases)
# print(death)
df= pd.DataFrame(districts)
case= pd.DataFrame(cases)
dead=pd.DataFrame(death)
df=pd.concat([df,case],axis=1)
df= pd.concat([df,dead],axis=1)
df.index+=1
df.columns=['District','Confirmed','Death']
print(df)