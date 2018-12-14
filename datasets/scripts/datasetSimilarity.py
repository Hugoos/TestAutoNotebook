from math import *

def squareRoot(x):

    return round(sqrt(sum([a*a for a in x])),3)

def cosineSimilarity(x,y,defaultValue = 0):

    longInput = {}
    shortInput = {}
    
    if len(x) > len(y):
        longInput = x
        shortInput = y
    else:
        shortInput = x
        longInput = y

    longVector = []
    shortVector = []

    for key in longInput.keys():
        if str(longInput[key]) != "nan":
            longVector.append(float(longInput[key]))
        elif key in shortInput and str(shortInput[key]) != "nan":
            longVector.append(float(defaultValue)) #Using default value as placeholder for nan
        if key in shortInput:
            if str(shortInput[key]) != "nan":
                shortVector.append(float(shortInput[key]))
            elif str(longInput[key]) != "nan":
                shortVector.append(float(defaultValue)) #Using default value as placeholder for nan
        else:
            shortVector.append(float(defaultValue)) #Using default value as placeholder for missing values
    numerator = sum(a*b for a,b in zip(shortVector,longVector))
    denominator = squareRoot(longVector)*squareRoot(shortVector)
    
    return round(numerator/float(denominator),3)
    
