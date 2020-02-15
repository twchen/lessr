# LESSR
A PyTorch implementation of LESSR (Lossless Edge-order preserving aggregation and Shortcut graph attention for Session-based Recommendation).

# Requirements
- CUDA 10.1
- Anaconda

# Usage
1. Create a conda environment with the required packages.
    ```sh
    conda env create -f packages.yml
    ```

2. Activate the created conda environment.
    ```
    conda activate lessr
    ```

3. Download and extract the datasets.
    - [Diginetica](https://cikm2016.cs.iupui.edu/cikm-cup/)
    - [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)
    - [Last.fm](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)

4. Preprocess the datasets using `preprocess.py`. Use the following command to see how to use `preprocess.py`.
    ```sh
    python preprocess.py -h
    ```

5. Train the model using `main.py`. Use the following command to see how to use `main.py`.
    ```sh
    python main.py -h
    ```

6. You can run the following command to train a model using a sample dataset.
    ```sh
    python main.py
    ```
