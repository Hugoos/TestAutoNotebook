# Automatic Jupyter Notebook

--THIS PROJECT HAS COME TO AN END, NO FURTHER UPDATES WILL COME ON THIS GITHUB PAGE--

This is a project about automatic generating Jupyter Notebook for OpenML dataset.
The program will pull an OpenML dataset from their servers using the datasetID.
The program will attempt to classify what kind of machine learning problem can be solved based on the target feature.
Currently only classification problems are supported, it will recognize regression problems but they currently don't work, additionally clustering problems are recognized but not officially supported as of yet.
Some information about the algorithms is displayed and hyperparameters can be adjusted within the notebook.

## Generating Jupyter notebook
The Jupyter Notebook can be automatic generated and run by the following command:
```
python auto-jupyter-notebook datasetID
```
It is possible to set a complexity threshold which will stop complex and long running algorithms from running by using the following command:
```
python auto-jupyter-notebook datasetID complexity
```
where complexity = {low,mid,high,inf} default = mid
passing inf will run all algorithms regardless of complexity.

If you want to generate multiple notebooks use one of the following commands:
```
python auto-jupyter-notebook [datasetID1,datasetID2,ect] 
python auto-jupyter-notebook [datasetID1,datasetID2,ect] complexity
```
Note that there must not be any spaces within the first argument and it must be formatted as a list.

The output file is called datasetID.ipynb 



