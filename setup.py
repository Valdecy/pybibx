from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pybibx',
    version='5.2.7',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pybibx',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'bertopic',
        'bert-extractive-summarizer',
        'chardet',
        'google-generativeai',
        'gensim',
        'keybert',
        'llmx',
        'matplotlib',
        'networkx',
        'numba',
        'numpy',
        'pandas',
        'Pillow',
        'plotly',
        'scipy',
        'scikit-learn',
        'sentencepiece',
        'sentence-transformers',
        'torch', 
        'torchvision',
        'torchaudio',
        'transformers',
        'umap-learn',
        'openai',
        'wordcloud'
    ],
    zip_safe=True,
    description='A Bibliometric and Scientometric Library Powered with Artificial Intelligence Tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
