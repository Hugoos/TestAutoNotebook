# -*- coding: utf-8 -*-
from strings.strings import *
import nbformat as nbf
import warnings
import os
import sys
import openml as oml
import ast
import shutil

warnings.simplefilter(action="ignore", category=RuntimeWarning)

nb = nbf.v4.new_notebook()

def main():
    # print command line arguments sys.argv[1:]:
    newpath = r'datasets'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    datasets = sys.argv[1]
    if datasets[0] == "[" and datasets[len(datasets)-1] == ",":
        print("Operation aborted")
        print("It appears you are trying to convert multiple datasets, please format the first argument as an array without spaces i.e. [1,100,1000]")
        return 0
    datasets = ast.literal_eval(datasets)
    if len(sys.argv) > 2:
        comp = sys.argv[2]
    else:
        comp = "default"
        
    complexity = calc_comp(comp)
    
    if isinstance(datasets, list):
        for dataset in datasets:
            print("Generating jupyter notebook for dataset "+str(dataset)+"...")
            generate_jnb(dataset, complexity)
    else:
        print("Generating jupyter notebook for dataset "+str(datasets)+"...")
        generate_jnb(datasets, complexity)
        
def calc_comp(comp):
    comp = comp.lower()
    if comp == "low":
        return 5000000000000
    elif comp == "mid":
        return 50000000000000
    elif comp == "high":
        return 5000000000000000
    elif comp == "inf":
        return -1
    else:
        return 50000000000000

def create_block(text, code):
    if isinstance(text, list):
        for t in text:
            nb['cells'].append(nbf.v4.new_markdown_cell(t))
    else:
        nb['cells'].append(nbf.v4.new_markdown_cell(text))
    if isinstance(code, list):
        for c in code:
            nb['cells'].append(nbf.v4.new_code_cell(c))
    else:
        nb['cells'].append(nbf.v4.new_code_cell(code))


def generate_jnb(dataset, complexity):
    nb['cells'] = []
    fname = "datasets/" + str(dataset)+ '.ipynb'
    ds = oml.datasets.get_dataset(dataset)
    text_title = """\
    # Automatic Jupyter Notebook for OpenML dataset %s: %s""" % (dataset,ds.name)
    create_block(text_title, code_library(dataset))
    nb['cells'].append(nbf.v4.new_code_cell(run_problemType))
    nb['cells'].append(nbf.v4.new_code_cell(run_similarity))
    create_block(text_comp, code_comp(str(complexity)))
    nb['cells'].append(nbf.v4.new_code_cell(run_featureImportance))
    nb['cells'].append(nbf.v4.new_code_cell(run_baselines))
    nb['cells'].append(nbf.v4.new_code_cell(run_algorithms))
    
    #create_block(text_landmarkers,code_get_landmarkers)
    #create_block(text_distances,[code_get_distances,code_compute_similar_datasets,code_get_datasets_name,code_landmarkers_plot,code_similarity_plot])
    
    # nb['cells'] = [nbf.v4.new_markdown_cell(text_title),
    # nbf.v4.new_markdown_cell(text_landmarkers),
    # nbf.v4.new_code_cell(code_get_landmarkers),
    # nbf.v4.new_markdown_cell(text_distances),
    # nbf.v4.new_code_cell(code_get_distances),
    # nbf.v4.new_code_cell(code_compute_similar_datasets),
    # nbf.v4.new_code_cell(code_get_datasets_name),
    # nbf.v4.new_code_cell(code_landmarkers_plot),
    # nbf.v4.new_code_cell(code_similarity_plot)]

    with open(fname, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    os.system("jupyter nbconvert --execute --inplace %s"%(fname))
    

if __name__ == "__main__":
    main()

