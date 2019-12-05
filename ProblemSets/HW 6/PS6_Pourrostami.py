Here I will get different data Using APIs and Web Scraping tools.
###########################################################################################################################
Part 1- Getting Data by Using API

# in this part I would like to see how GDP per capita has changed overtime for 4 Asian 
# countries relative to the United States and the world level of GDP per capita from 1960 to 2019. 
# I choose Japan, China, India and South Korea as they are sited as top economies in Asia.
# The data is comming from World Bank


# import pandas-datareader and other packages
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import wb
%matplotlib inline
%matplotlib notebook

ind = "NY.GDP.PCAP.CD"
cons = ["CHN", "IND", "USA", "JPN", "KR", "WLD"]
st = 1960
ed = 2019
data = wb.download(indicator=ind, country=cons, start=st, end=ed).dropna()

# Renaming the variable names
data = data.rename(columns={'NY.GDP.PCAP.CD':'GDP perCapita'})

#data.reset_index(inplace=True)
#data=data.pivot(index='year', columns='country', values='GDP perCapita')
    
data.head()
#data is "pivoted", pandas' unstack fucntion helps reshape it into something plottable
data1 = data.unstack(level=0)
data1.sort_index().plot()  #you can change x 
plt.legend(loc=2,bbox_to_anchor=(1.0,1.0)) #include a legend
plt.title("Distribution of GDP per Capita") 
plt.xlabel('year'); plt.ylabel('Thousands of US Dollars')
data1.plot()
plt.show()
plt.savefig('Distribution of GDP per Capita.png')

##########################################################################################################################
Part 2- Getting Data by Web Scraping

# import packages
from bs4 import BeautifulSoup
import urllib.request

# give URL and header
wiki = "https://en.wikipedia.org/wiki/List_of_international_goals_scored_by_Ali_Daei"
header = {'User-Agent': 'Google Chrome'} #Needed to prevent 403 error on Wikipedia

# Make the request to get served the webpage, "soupify" it
req = urllib.request.Request(wiki, headers=header)
page = urllib.request.urlopen(req)
soup = BeautifulSoup(page, 'lxml')
 
# what does the soup object contain
print(soup.prettify())
# print(page)

# extract the table by pulling information from the wikitable class
table = soup.find_all("table", {"class": "wikitable"})[0] # Grab the first table
print(table)

# containing the element in each row in that column
goals = {'#':[], 'Date': [], 'Venue': [], 'Opponent': [],'Score': [], 'Result': [], 'Competition': []}

# iterate through the table, pulling out each row

for row in table.findAll("tr"):
    cells = row.findAll("td")
    print(cells)
    #For each "tr", assign each "td" to a variable.
    if len(cells) == 7:
        goals['#'].append(cells[0].findAll(text=True))
        goals['Date'].append(cells[1].findAll(text=True))
        goals['Venue'].append(cells[2].findAll(text=True))
        goals['Opponent'].append(cells[3].find(text=True))
        goals['Score'].append(cells[4].find(text=True))
        goals['Result'].append(cells[5].find(text=True))
        goals['Competition'].append(cells[6].findAll(text=True))
# Look at this dictionary
goals

# put this in a dataframe and format it
import pandas as pd

goals_data = pd.DataFrame(goals)
goals_data

# this table shows the information about the games that Ali played in them.

df= goals_data.drop(['#', 'Opponent'], axis=1)
df

df = df.astype('str')
df.Date = df.Date.str.strip('[').str.strip(']').str.strip("''")
df.Venue = df.Venue.str.strip('[').str.strip(']').str.strip("''")
df.Score = df.Score.str.strip('[').str.strip(']').str.strip("''")
df.Result = df.Result.str.strip("''")
df.Result = df.Result.str.strip("''")
df.Competition = df.Competition.str.strip('[').str.strip(']').str.strip("''").str.strip("'").str.strip(",").str.strip(r'\n')
print(df)

df.to_excel("output.xlsx") 

# extract the table by pulling information from the wikitable class
# I am trying to get data from second table
# this table will show his goals by year
table = soup.find_all("table", {"class": "wikitable"})[1] # Grab the second table
print(table)

# containing the element in each row in that column
goals_by_year = {'Year':[], 'Apps': [], 'Goals': []}

# iterate through the table, pulling out each row

for row in table.findAll("tr"):
    cells = row.findAll("td")
    print(cells)
    #For each "tr", assign each "td" to a variable.
    if len(cells) == 3:
        goals_by_year['Year'].append(cells[0].findAll(text=True))
        goals_by_year['Apps'].append(cells[1].findAll(text=True))
        goals_by_year['Goals'].append(cells[2].findAll(text=True))
        
# Look at this dictionary

goals_by_year
        
# put this in a dataframe and format it
import pandas as pd

goals_by_year_data = pd.DataFrame(goals_by_year)

goals_by_year_data

# this table will show his goals by year        

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

goals_by_year_data = goals_by_year_data.astype('str')
# goals_by_year_data.replace(to_replace=[r"['", r"']",r"\n"], value=None, regex=True, inplace=True)#.astype(float)
goals_by_year_data.Goals = goals_by_year_data.Goals.str.strip('[').str.strip(']').str.strip("''").str.strip(r'\n')
goals_by_year_data.Year = goals_by_year_data.Year.str.strip('[').str.strip(']').str.strip("''") 
goals_by_year_data.Apps = goals_by_year_data.Apps.str.strip('[').str.strip(']').str.strip("''")
goals_by_year_data = goals_by_year_data.astype('float')
goals_by_year_data.to_excel("output1.xlsx")  
print(goals_by_year_data)

# I am trying to get data from forth table
# # this table will show his goals by opposition
table = soup.find_all("table", {"class": "wikitable"})[3] # Grab the second table
print(table)

# containing the element in each row in that column
goals_by_opposition = {'Opposition':[], 'Goals': []}

# iterate through the table, pulling out each row

for row in table.findAll("tr"):
    cells = row.findAll("td")
    print(cells)
    #For each "tr", assign each "td" to a variable.
    if len(cells) == 2:
        goals_by_opposition['Opposition'].append(cells[0].findAll(text=True))
        goals_by_opposition['Goals'].append(cells[1].findAll(text=True))
        
# Look at this dictionary
goals_by_opposition


# put this in a dataframe and format it
import pandas as pd

goals_by_opposition_data = pd.DataFrame(goals_by_opposition)
goals_by_opposition_data
 
## this table will show his goals by opposition

goals_by_opposition_data = goals_by_opposition_data.astype('str')
goals_by_opposition_data.Opposition = goals_by_opposition_data.Opposition.str.strip('[').str.strip(']').str.strip("''").str.strip(r'\xa0').str.strip(",").str.strip("''")
goals_by_opposition_data.Goals = goals_by_opposition_data.Goals.str.strip('[').str.strip(']').str.strip("''").str.strip(r'\n')
goals_by_opposition_data.to_excel("output2.xlsx") 
print(goals_by_opposition_data)







