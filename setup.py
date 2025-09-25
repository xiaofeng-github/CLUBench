from setuptools import setup, find_packages
from os import path as osp

# read the contents of requirements.txt
with open(osp.join(osp.dirname(osp.abspath(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="CLUBench",
    version='0.1',
    author="",
    author_email="",
    url='',
    description='Python package for Clustering',
    long_description='Python package for Clustering: 131 benchmark datasets are collected including image, text and tabular;',
    packages=find_packages(),
    include_package_data=False,
    install_requires=requirements,
    keywords=['clustering', 'benchmark']
)