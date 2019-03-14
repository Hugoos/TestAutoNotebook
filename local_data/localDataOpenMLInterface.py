import ast
import operator

import numpy as np
import openml as oml
import pandas as pd
from datasetSimilarity import *


def initializeDatasets(name, initContents):
    f = open('../local_data/' + name + '.txt','w')
    #if not f.readline():
    print("creating new " + name + " file")
    f.write(str(initContents))
    f.seek(0)
    return f

def writeDatasets(name, contents):
    f = open('../local_data/' + name + '.txt','w')
    print("writing to " + name + " file")
    f.write(str(contents))
    f.close()

def findFeatureRanges():
    initializeDatasets("metaFeaturesByType", "{}")
    initializeDatasets("metaFeaturesMetadata", "{}")
    features = {}
    featuresMetaData = {}
    f = open('../local_data/datasetMetaFeatures.txt','a+')
    f.seek(0)
    metaFeatures = eval(f.readline())
    for key in metaFeatures.keys():
        for key2 in metaFeatures[key].keys():
            if key2 not in features:
                features[key2] = []
            if str(metaFeatures[key][key2]) != "nan":
                features[key2].append(metaFeatures[key][key2])
    for key in features.keys():
        featuresMetaData[key] = {}
        feature = features[key]
        feature.sort()
        featuresMetaData[key]["minimum"] = feature[0]
        featuresMetaData[key]["maximum"] = feature[-1]
        featuresMetaData[key]["mean"] = np.mean(feature)
        featuresMetaData[key]["median"] = np.median(feature)
        featuresMetaData[key]["std"] = np.std(feature)
        features[key] = feature
        
    writeDatasets("metaFeaturesByType", features)
    writeDatasets("metaFeaturesMetadata", featuresMetaData)

def normalizeMetaFeatures():
    normalizedMetaFeatures = {}
    initializeDatasets("datasetMetaFeaturesNormalized", "{}")
    f = open('../local_data/datasetMetaFeatures.txt','r')
    metaFeatures = eval(f.readline())
    f2 = open('../local_data/metaFeaturesMetadata.txt','r')
    metaFeaturesMetaData = eval(f2.readline())
    for did in metaFeatures.keys():
        normalizedMetaFeatures[did] = {}
        for quality in metaFeatures[did].keys():
            x = metaFeatures[did][quality]
            xp = (x - metaFeaturesMetaData[quality]['mean']) / metaFeaturesMetaData[quality]['std']
            normalizedMetaFeatures[did][quality] = xp
    writeDatasets("datasetMetaFeaturesNormalized", normalizedMetaFeatures)

def createUpdatedSimilarityMatrix(name, features, defaultValue = 0):
    updatedSimilarityMatrixf = initializeDatasets(name, "{}")
    updatedSimilarityMatrix = {}
    f = open('../local_data/' + features + '.txt','r')
    metaFeatures = eval(f.readline())
    f.close()
    f = open('../local_data/datasetSimilarityMatrix.txt','r')
    oldSimilarityMatrix = eval(f.readline())
    f.close()

    for did in oldSimilarityMatrix.keys():
        for did2 in oldSimilarityMatrix[did].keys():
            if did not in updatedSimilarityMatrix.keys():  
                updatedSimilarityMatrix[did] = {}
            if did2 not in updatedSimilarityMatrix.keys():  
                updatedSimilarityMatrix[did2] = {}
            if did != did2:
                sim = cosineSimilarity(metaFeatures[did], metaFeatures[did2], defaultValue)               
                updatedSimilarityMatrix[did][did2] = sim
                updatedSimilarityMatrix[did2][did] = sim                
            else:
                updatedSimilarityMatrix[did][did2] = -1
    writeDatasets(name, updatedSimilarityMatrix)

def findTopNSimilarDatasets(simMatrix, did, N):
    f = open('../local_data/' + simMatrix + '.txt','r')
    simMatrix = eval(f.readline())
    f.close()
    scores = simMatrix[did]
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_scores[:N]

def showTopNSimilarDatasets(simMatrix, did, N):
    datasets = findTopNSimilarDatasets(simMatrix, did, N)
    datasetsOML = oml.datasets.list_datasets()
    newDatasetsDict = []
    for did,csim in datasets:
        newDatasetsDict.append({'did': did, 'name': datasetsOML[did]['name'], 'similarity': csim, 'url': "https://www.openml.org/d/" + str(did)})
    df = pd.DataFrame.from_dict(newDatasetsDict)
    df.style.format({'url': make_clickable})
    print(df)

def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val)


    
def updateLocalDatasets(datasets):
    newDatasetsList = []
    indexf = initializeDatasets("datasetIndex", "[]")
    metaFeaturef = initializeDatasets("datasetMetaFeatures", "{}")
    similarityMatrixf = initializeDatasets("datasetSimilarityMatrix", "{}")
    
    index = []
    metaFeatures = {}
    similarityMatrix = {}
    
    indexf.close()
    metaFeaturef.close()
    similarityMatrixf.close()
    
    for key in datasets.keys():
        if key not in index:
            newDatasetsList.append(key)
    #print(newDatasetsList)
    for did in newDatasetsList:
        similarityMatrix[did] = {}
        try:
            data = oml.datasets.get_dataset(did)
            metaFeatures[did] = data.qualities
            for key in metaFeatures.keys():
                if key != did:
                    sim = cosineSimilarity(metaFeatures[did], metaFeatures[key], 0)
                    similarityMatrix[did][key] = sim
                    similarityMatrix[key][did] = sim
                else:
                    similarityMatrix[did][did] = -1
            index.append(did)
        except:
            print("Dataset " + str(did) + " failed to be processed correctly")
    writeDatasets("datasetMetaFeatures", metaFeatures)
    writeDatasets("datasetSimilarityMatrix", similarityMatrix)
    writeDatasets("datasetIndex", index)

def generateLocalData():
    updateLocalDatasets(oml.datasets.list_datasets()) #Create datasetIndex, datasetMetaFeatures, datasetSimilarityMatrix
    findFeatureRanges() #Create metaFeaturesByType metaFeaturesMetadata
    normalizeMetaFeatures() #Create datasetMetaFeaturesNormalized
    createUpdatedSimilarityMatrix("datasetSimilarityMatrixNormalized", "datasetMetaFeaturesNormalized", defaultValue = 0) #Create datasetSimilarityMatrixNormalized datasetMetaFeaturesNormalized
    
    
