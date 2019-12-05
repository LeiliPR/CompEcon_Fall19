# first I import packages I will need

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

# import the dataset
# the numbers in data are in billion Rial
dt = pd.read_excel('exchange rate.xlsx')
dt.head()

###################################################Part 1#####################################################
# First Visualization

lrer = dt['LRER']   
year = dt['year']  

# Investigating Iran's Real Exchange rate Trend During the Years of 1979 and 2016
plt.rcParams.update({'font.size': 15})
fig = plt.figure(figsize=(20,10))
plt.style.use('ggplot') # style (theme) for plot
fig, ax = plt.subplots() # make figure and axes separate objects
x = np.array(year)
y = np.array(lrer)
plt.plot(x, y, 'go--')
ax.set(title='Real Exchange Rate of Iran 1979-2016', xlabel='Year',
       ylabel="Real Exchange Rate") # plot title, axis labels
ax.axvline(x=1992, color='k', linestyle='--') #insert vertical line at year 1992
ax.axvline(x=2002, color='k', linestyle='--') #insert vertical line at year 2002
ax.axvline(x=2013, color='k', linestyle='--') #insert vertical line at year 2013
# save figure
fig.savefig('Real Exchange Rate.png')

# Second Visualization

tot = dt['TOT']   
year = dt['year']

# Investigating the Trend of terms of trade of Iran during Years 1979-2016
# Bar plot 
plt.style.use('ggplot') # select a style (theme) for plot
plt.bar(x, y, alpha=0.5)
x = np.array(year)
y = np.array(tot)
plt.ylabel('Terms of Trade')
plt.xlabel('Year')
plt.title('Terms of Trade of Iran 1979-2016')
fig.savefig('TOT.png')

# third Visualization

officialER = dt['EROff']
nonofficialER = dt['ERNONOFF']
year = dt['year']
#Examination of the official exchange rate trend and the non official exchange rate trend over the years 1979-2016

plt.rcParams.update({'font.size': 15})
fig = plt.figure(figsize=(20,10))
plt.style.use('ggplot') # style (theme) for plot
fig, ax = plt.subplots() # make figure and axes separate objects
x = np.array(year)
y1 = np.array(officialER)
y2 = np.array(nonofficialER)
plt.plot(x, y1, axes=ax)
plt.plot(x, y2, axes=ax)
chart_names = [' Exchange Rates,USD/IRR,official Rate', 'Exchange Rates,USD/RR,nonofficial Rate']

# plt.xticks(np.round_(bin_cuts, 1))
plt.title('Official and Non Official Exchange Rate', fontsize=17)
plt.xlabel('Year')

plt.legend(labels=chart_names,loc=2,bbox_to_anchor=(1.0,1.0)) #include a legend

# Forth Visualization

#investigating the Relationship Between Real Exchange Rate Misalingment and Iranian Gross Domestic Product.
erm = dt['Exchange rate Misalingment']
lgdp = dt['LGDP']

plt.rcParams.update({'font.size': 15})
fig = plt.figure(figsize=(20,10))
plt.style.use('ggplot') # style (theme) for plot
fig, ax = plt.subplots() # make figure and axes separate objects
x = np.array(lgdp)
y = np.array(erm)
               
plt.scatter(x, y, alpha=0.50)
plot_names = ['Correlation between GDP and Exchange Rate Misalingment']

#plt.xticks(np.round_(bin_cuts, 1))
plt.title('Exchange Rate Misalingment and Growth', fontsize=17)
plt.xlabel('LGDP')
plt.ylabel('ERM')

fig.savefig('ERM-GDP.png')

####################################################Part 2######################################

 # import Python packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linearmodels as lm
from stargazer.stargazer import Stargazer


# To obtain the equilibrium exchange rate and calculate the real exchange rate Misalignment, I estimate the equation (3) by the OLS method for the annual data of 1979-2016
# Define the model
lrer_ols = smf.ols(formula='LRER ~ LOPN + LOIL + LESUB', data = dt)
# Estimate the model
res1 = lrer_ols.fit()
# Show the results
print(res1.summary())


# and I place them in the following model which is modification form of Wong (2013)’s model in his studies: 
# I drove GDP per Capita by deviding nominal GDP on population of Iran
# LI is Logarithm of the ratio of the gross fixed capital formation to gross domestic product

lpgdp_ols = smf.ols(formula='LPGDP ~ LI + LTOT + PO + NE', data = dt)
res2 = lpgdp_ols.fit()
# Show the results
print(res2.summary())





