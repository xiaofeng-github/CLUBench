![](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=Python&logoColor=ffffff)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# CLUBench (A Clustering Benchmark).

![RoadMap of CLUBench](./figs/MAP.png)


## *Abstract*
Clustering is a fundamental problem in data science with a long-standing research history. Over the past decades, numerous clustering algorithms, ranging from conventional machine learning approaches to deep clustering methods, have been developed. Despite this progress, a systematic and large-scale empirical evaluation that jointly considers conventional algorithms, deep learning-based methods and recent foundation model-based clustering remains largely absent, leading to limited guidance on algorithm selection and deployment. To address this gap, we introduce CLUBench, a comprehensive clustering benchmark comprising 24 algorithms of diverse principles evaluated on 131 datasets across tabular, text and image data. Importantly, CLUBench provides a unified comparison between state-of-the-art baselines and foundation model-energized clustering strategies on all three modalities (tabular, text and image). Extensive experiments (178,815) in CLUBench yield statistically meaningful insights and identify promising yet underexplored pathways about clustering research. For example, we observe low-rank structure in cross-model performance matrices, which facilitates an efficient strategy for rapid algorithm evaluation and selection in practical applications. In addition, we provide an easy-to-use toolbox by encapsulating the source codes from the official code repository into a unified framework, accompanied by detailed instructions.

## **Prerequisites**
1. ### Environment:
    - Python 3.10


2. ### Datasets: due to space limitation in GitHub, we cannot upload the whole 131 Benchmark Datasets, only 10 datasets are provided in /CLUBench/datasets. The overall datasets can be download in [CLUBench-Datasets](https://huggingface.co/datasets/Feng-001/Clustering-Benchmark).

3. ### Install CLUBench

    - `pip install -e .`


## **Usages**

- ### Complete Benchmark Datasets

    ```
    from CLUBench import DATASETS

    print('CLUBench Datasets =================================================')
    print(DATASETS)
    ```

- ### Clustering (e.g. using DEC) on benchmark dataset (e.g. weather.npz)

    ```
    from CLUBench import DEC, load_data
    
    data_name = 'weather.npz'
    X, Y = load_data(data_name)

    hpc = {
        'n_clusters': len(np.unique(Y))
        # you can set more hyperparameters here.
    }
    CM = DEC(**hpc)
    CM.fit_predict(X)
    acc, nmi, ari = CM.evaluation(Y)

    print(f'acc: [{acc:.4f}]')
    print(f'nmi: [{nmi:.4f}]')
    print(f'ari: [{ari:.4f}]')
    print(f'time: [{CM.time:.4f}]')

    ```

- ### Clustering using predefined HPC

    ```
    from CLUBench import DEC, load_data, load_hpc
    
    data_name = 'weather.npz'
    X, Y = load_data(data_name)

    # load predefined hyperparameter configuration of 'DEC'
    hpc = load_hpc(hpc_name='DEC')

    # you can tune the model hyperparameter configuration by changing or creating json file in ./CLUBench/hpc.
    
    hpc.update({'n_clusters': len(np.unique(Y))})

    CM = DEC(**hpc)
    CM.fit_predict(X)
    acc, nmi, ari = CM.evaluation(Y)

    print(f'acc: [{acc:.4f}]')
    print(f'nmi: [{nmi:.4f}]')
    print(f'ari: [{ari:.4f}]')
    print(f'time: [{CM.time:.4f}]')

    ```


## **Extending New Datasets**

### Step 1. Constructing the data dict {'x': data, 'y': labels}, type(data) == (list or numpy)
### Step 2. Saving as binary file ('data_name'.npz) in ./CLUBench/datasets
### Step 3. Adding the dataset_name into 'DATASETS' list in ./CLUBench/configs.py


## **Extending New algorithms**

### Step 1. Creating a python file ('new_algorithm.py') in ./CLUBench/algorithms 
### Step 2. Instancing a new Class (NewAlgo) inheriting the BaseCluster and implementing the abstract function fit_predict(self, X)
### Step 3. import the 'NewAlgo' in the __init__.py in ./CLUBench/


## **Low-Rank Analysis**


## **Model Selection**


## **Performance Matrices**