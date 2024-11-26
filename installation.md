
0. Download the repo using the below command 
```
git clone https://github.com/shekharlab/SpatialRGC.git
```

1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

This will modify the `PATH` variable in your bashrc. Open a new terminal for that path change to take place (otherwise you won't be able to find conda' in the next step).

2. Create a conda environment that will contain python 3:
```
conda create -n spatial_rgc python=3.10.4
```

3. activate the environment (do this every time you open a new terminal and want to run code):
```
source activate spatial_RGC
```

4. Install the requirements into this conda environment
```
pip install -r requirements.txt
```

5. Allow your code to be able to see the 'spatial_rgc' package installed by setup.py
```
cd <path_to_this directory>
pip install -e .
```

6. Download necessary .h5ad data files from Zenodo. Note that raw data will be hosted soon, but can be provided by emailing the authors.
