![](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=Python&logoColor=ffffff)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# CLUBench (A Clustering Benchmark).

![](./figs/MAP.png)


## *Abstract*
Clustering is a fundamental problem in data science with a long-standing research history. Over the past few decades, numerous clustering algorithms have been developed. However, a systematic and experimental evaluation of these algorithms remains lacking and is urgently needed. To address this gap, we introduce CLUBench, a comprehensive clustering benchmark comprising 23 algorithms of diverse principles evaluated on 131 datasets across tabular, text and image data types. Our extensive experiments (174,485) yield statistically meaningful insights into the performance of various clustering methods, such as the impact of hyperparameter tuning, similarity between algorithms, and the impact of data type and dimension.
Notably, we observe low-rank characteristics in cross-model performance matrices, which facilitates an efficient strategy for rapid algorithm evaluation and selection in practical applications. Additionally, we provide an easy-to-use toolbox by encapsulating the source codes from the official code repository into a unified framework, accompanied by detailed instructions. With CLUBench, researchers and practitioners can efficiently select appropriate algorithms or datasets for evaluating new datasets or proposed methods.

## **Prerequisites**
1. ### Environment:
    - Python 3.10


2. ### Datasets: due to space limitation in GitHub, we cannot upload the whole 131 Benchmark Datasets, now only 10 datasets are provided in /CLUBench/datasets. We will provide a download link after review.

3. ### Install CLUBench

    - `pip install -e .`


## **Usages**

- ### Clustering (e.g. using DSCN) on benchmark dataset (e.g. weather.npz)

    ```
    import numpy as np
    from CLUBench import DSCN, load_data
    
    data_name = 'weather.npz'
    X, Y = load_data(data_name)

    hpc = {
        'n_clusters': len(np.unique(Y))
        # you can use more hyperparameters here.
    }
    CM = DSCN(**hpc)
    CM.fit_predict(X)
    acc, nmi, ari, time = CM.evaluation(Y)

    print(f'acc: [{acc:.4f}]')
    print(f'nmi: [{nmi:.4f}]')
    print(f'ari: [{ari:.4f}]')
    print(f'time: [{time:.4f}]')

    ```
