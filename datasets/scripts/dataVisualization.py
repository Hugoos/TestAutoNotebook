#%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd


def show1DHist(data):
    X, features = data.get_data(return_attribute_names=True)
    df = pd.DataFrame(X, columns=features)
    df.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
            xlabelsize=8, ylabelsize=8, grid=False, figsize=(16,9))
    plt.tight_layout()
