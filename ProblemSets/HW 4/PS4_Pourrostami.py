##############################################################################################################
# First I import packages that I will use

import pandas as pd
import numpy as np
import math
import scipy.optimize as opt
from scipy.optimize import minimize
from geopy.distance import vincenty

# Data
data=pd.read_excel('radio_merger_data.xlsx')
data.describe()

# we have price in millions of dollar  
data['priceMIL'] = data['price']/1000000

# we have population in millions of numbers 
data['populationMIL'] = data['population_target']/1000000

##################################################Part One####################################################

# I must set up the data in one array for use in the score function,

# to do so, I define the years and then I create an empty array that will hold all observations and counterfactuals

# for preparing the data for actual and counterfactual matches I devide the data set into 2

years1 = len(data[(data['year']<2008)])
years2 = len(data[(data['year']>2007)])

# now I want to create the observations row by row

# we have 4 equation each with 3 terms so our final matrix should be 2421 * 12

# the equation is :::::      "fm(b,t) = x1bmy1tm + αx2bmy1tm + βdistancebtm + epsilonbtm"

m = 1  
bt = 1
years = years1
arrayone = np.empty((0, 12))
while (m <= 2):
    while (bt <= years-1):
        K = 1
        while (K <= years-bt):
            p1 = (data.iloc[bt-1, 3], data.iloc[bt-1, 4])
            p2 = (data.iloc[bt-1, 5], data.iloc[bt-1, 6])
            p3 = (data.iloc[bt+K-1, 3], data.iloc[bt+K-1, 4])
            p4 = (data.iloc[bt+K-1, 5], data.iloc[bt+K-1, 6])

            x1bmy1tm = data.iloc[bt-1, 9] * data.iloc[bt-1, 12]   #f(b,t)
            x2bmy2tm = data.iloc[bt-1, 11] * data.iloc[bt-1, 12]
            distbtm = vincenty(p1, p2).miles
            

            x1qmy1um = data.iloc[bt+K-1, 9] * data.iloc[bt+K-1, 12]   #f(b',t')
            x2qmy1um = data.iloc[bt+K-1, 11] * data.iloc[bt+K-1, 12]
            distqu = vincenty(p3, p4).miles

            x1bmy1um = data.iloc[bt-1, 9] * data.iloc[bt+K-1, 12]  #f(b',t')   
            x2bmy1um = data.iloc[bt-1, 11] * data.iloc[bt+K-1, 12]
            distbu = vincenty(p1, p4).miles

            x1qmy1tm = data.iloc[bt+K-1, 9] * data.iloc[bt-1, 12]   #f(b',t')
            x2qmy1tm = data.iloc[bt+K-1, 11] * data.iloc[bt-1, 12]
            distqt = vincenty(p3, p2).miles
            
            # I put the observations in the result array
            
            dataforradio = np.array([x1bmy1tm, x2bmy2tm, distbtm, x1qmy1um, x2qmy1um, distqu, x1bmy1um, x2bmy1um, distbu, x1qmy1tm, x2qmy1tm, distqt])
            
            K = K + 1
        
            arrayone = np.append(arrayone, [dataforradio], axis=0)

        bt = bt + 1
    years = years1 + years2 - 1
    m = m + 1

print(arrayone)

arrayone.shape

##################################################################################################################

# score function

def mse(prms_1, arrayone):
    alpha, beta = prms_1
    sum = 0
    i = 0
    while(i <= len(arrayone)-1):
        f_bt_A = arrayone[i, 0] + alpha * arrayone[i, 1] + beta * arrayone[i, 2] + arrayone[i, 3] + alpha * arrayone[i, 4] + beta * arrayone[i, 5]           
        f_bt_B = arrayone[i, 6] + alpha * arrayone[i, 7] + beta * arrayone[i, 8] + arrayone[i, 9] + alpha * arrayone[i, 10] + beta * arrayone[i, 11]
        
        if f_bt_A > f_bt_B :
            sum = sum - 1

        i = i + 1
        print(sum)
    return sum

# initial guess
b1 = (0.5, -0.5)

# optimization
f_bt1_ = opt.minimize(mse, b1, arrayone, method = 'Nelder-Mead', options={'disp': True})

print(f_bt1_)

################################################Part Two############################################################

# in this part I include prices and the HHI index 

# again I want to create the observations row by row

# we have 4 equation each with 5 terms so our final matrix should be 2421 * 20

# the equation is ::::::  "fm(b,t) = δx1bmy1tm + αx2bmy1tm + γHHItm + βdistancebtm + epsilonbtm"

years1 = len(data[(data['year']<2008)])
years2 = len(data[(data['year']>2007)])

m = 1  
bt = 1
years = years1
arraytwo = np.empty((0, 20))

while (m <= 2):
    while (bt <= years-1):
        K = 1
        while (K <= years-bt):
            p1 = (data.iloc[bt-1, 3], data.iloc[bt-1, 4])
            p2 = (data.iloc[bt-1, 5], data.iloc[bt-1, 6])
            p3 = (data.iloc[bt+K-1, 3], data.iloc[bt+K-1, 4])
            p4 = (data.iloc[bt+K-1, 5], data.iloc[bt+K-1, 6])

            x1bmy1tm = data.iloc[bt-1, 9] * data.iloc[bt-1, 12]   #f(b,t)
            x2bmy2tm = data.iloc[bt-1, 11] * data.iloc[bt-1, 12]
            hhibtm = data.iloc[bt-1, 8]
            distbtm = vincenty(p1, p2).miles
            

            x1qmy1um = data.iloc[bt+K-1, 9] * data.iloc[bt+K-1, 12]   #f(b',t')
            x2qmy1um = data.iloc[bt+K-1, 11] * data.iloc[bt+K-1, 12]
            hhiqu = data.iloc[bt-1, 8]
            distqu = vincenty(p3, p4).miles

            x1bmy1um = data.iloc[bt-1, 9] * data.iloc[bt+K-1, 12]  #f(b',t')   
            x2bmy1um = data.iloc[bt-1, 11] * data.iloc[bt+K-1, 12]
            hhibu = data.iloc[bt+K-1, 8]
            distbu = vincenty(p1, p4).miles

            x1qmy1tm = data.iloc[bt+K-1, 9] * data.iloc[bt-1, 12]   #f(b',t')
            x2qmy1tm = data.iloc[bt+K-1, 11] * data.iloc[bt-1, 12]
            hhigt = data.iloc[bt+K-1, 8]
            distqt = vincenty(p3, p2).miles
            
            price_11 = data.iloc[bt-1, 13]
            price_22 = data.iloc[bt+K-1, 13]

            dataforradio = np.array([x1bmy1tm, x2bmy2tm, hhibtm, distbtm, x1qmy1um, x2qmy1um, hhiqu, distqu, price_11, price_22, x1bmy1um, x2bmy1um, hhibu, distbu, x1qmy1tm, x2qmy1tm, hhigt, distqt, price_11, price_22])
            
            K = K + 1
            
# I put the observations in the result array
            
            arraytwo = np.append(arraytwo, [dataforradio], axis=0)

        bt = bt + 1
    years = years1 + years2 - 1

    m = m + 1

print(arraytwo)

arraytwo.shape

##############################################################################################################

#score function

def mse2(prms_2, arraytwo):
    delta, alpha, gamma, beta = prms_2
    sum = 0
    i = 0
    while(i <= len(arraytwo)-1):
        f_bt_A = delta * arraytwo[i, 0] + alpha * arraytwo[i, 1] + gamma * arraytwo[i, 2] + beta * arraytwo[i, 3] + arraytwo[i, 9] - delta * arraytwo[i, 10] - alpha * arraytwo[i, 11] - gamma * arraytwo[i, 12] - beta * arraytwo[i, 13] - arraytwo[i, 18]
    
        f_bt_B = delta * arraytwo[i, 4] + alpha * arraytwo[i, 5] + gamma * arraytwo[i, 6] + beta * arraytwo[i, 7] + arraytwo[i, 8] - delta * arraytwo[i, 14] - alpha * arraytwo[i, 15] - gamma * arraytwo[i, 16] - beta * arraytwo[i, 17] - arraytwo[i, 19]
        
     
        if f_bt_A > f_bt_B :
            sum = sum - 1

        i = i + 1
        
        print(sum)
        
    return sum

"Initial Guess"
b1 = (0.5, 1, 1, -0.5 )

"Optimization routine"
f_bt_2 = opt.minimize(mse2, b1, arraytwo, method = 'Nelder-Mead', options={'disp': True})

print(f_bt_2)


############################################################################################################