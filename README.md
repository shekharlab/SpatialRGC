# Code for "Spatial transcriptomic analysis of ganglion cell diversity and topography on retinal flatmounts"

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Cite](#cite)

## Project Overview
This repository contains analyses for the paper "Spatial transcriptomic analysis of ganglion cell diversity and topography on retinal flatmounts". 

The paper can be accessed here: 

[insert biorxiv link]

## Repository Structure
- **spatial_rgc/notebook_scripts/**:
    - **train_models.ipynb** Trains cell type classifiers on reference scRNA-seq datasets
    - **preprocess_data.ipynb**: Preprocesses raw cell by gene matrices
    - **mosaic_analysis.ipynb**: Runs mosaic analysis and creates associated figure (Figure S7)
    - **make_figures.ipynb**: Creates all spatial transcriptomics figures aside from the mosaic analysis (Figure S7).
-**spatial_rgc/utils/**: Contains many util scripts used by the notebooks
- **spatial_rgc/models/**: Contains models used for cell type classification
- **spatial_rgc/imaging_scripts/**: Scripts related to segmentation.
    - **images/**: [From hosted data; WIP] Raw image files
    - **mappings/**: [From hosted data; WIP] Coordinate mapping for mapping coordinates in cell by gene matrices to image files
    - **trials.sh** Run this to execute segmentation pipeline
    - **run_pipeline.sh** and **stitch_and_assign.sh**: Executes segmentation, stitching/transcripts, and adding antibody stains
    - **merge_masks.py, parallel_cell_matrix.py** and **add_costains.py** Segments cells,Assigns transcripts to cells, and adds antibody stains, respectively
    - **outputs/**: [Needs to be created manually] Output of segmentation pipeline
- **spatial_rgc/figures/**: [Needs to be manually created] Figures are saved to here
    - **Figure_1**: Files for Figure 1
    - ... etc
- **spatial_rgc/data/**: Data for each retina tissue section (generated by segmentatino pipeline) [Needs to be created and populated]

## Getting Started

1. See "installation.md" to clone this repository and install dependencies. Then read "spatial_rgc/file_structure.md" and ensure raw data is populated properly.
2. (Skip if not interested in recreating cell by gene matrix from raw images) Run **spatial_rgc/imaging_scripts/trials.sh** for 
3. Run through all analysis in **spatial_rgc/notebook_scripts/** in the order listed in Repository Structure to generate the figures

## Usage
Due to the size of the files, the data directory is empty by default. For running through analyses that only require the cell by gene matrix, see Zenodo. For now, please email [Nicole Tsai](mailto:Nicole.Tsai@ucsf.edu), [Kushal Nimkar](mailto:kushalnimkar@berkeley.edu), [Karthik Shekhar](mailto:kshekhar@berkeley.edu),or [Xin Duan](mailto:Xin.Duan@ucsf.edu) for raw image data from MERFISH. This raw data will be hosted separately soon.

## Cite
If you find our code, analysis, or results useful and use them in your publications, please cite us using the following citation: 

Tsai, Nimkar, ... Shekhar,Duan. Spatial transcriptomic analysis of ganglion cell diversity and topography on retinal flatmounts. *In submission*. 2024. 
