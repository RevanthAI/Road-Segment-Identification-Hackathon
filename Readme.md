# Road Segment Identification Hackathon 
## 4th Position solution Overview

The final Solution is an ensemble of two models EfficientnetB3(5 folds) and EfficientnetB5(3 folds and the 3rd fold was trained untill 9 epochs , as i couldnt train untill 5 folds due to time constraints) and the final predictions are an power average of those two models , these two models were trained using gradient centralization as they boosted the score with in less number of epoches and i have tried different ensemble techniques out of those power average ensemble worked better



## Data Conversion

The image were converted into jpeg format as the tensorflow dosent natively support decoding .tif files 

## Dependencies

Python 3.7.11 was used and the virtual environment is an anaconda environment the following dependencies are listed below

    Name                    Version                   Build  Channel
    _libgcc_mutex             0.1                        main  
    _openmp_mutex             4.5                       1_gnu  
    _tflow_select             2.1.0                       gpu  
    absl-py                   0.13.0           py37h06a4308_0  
    aiohttp                   3.7.4            py37h27cfd23_1  
    argcomplete               1.12.3                   pypi_0    pypi
    astor                     0.8.1            py37h06a4308_0  
    astunparse                1.6.3                      py_0  
    async-timeout             3.0.1            py37h06a4308_0  
    attrs                     21.2.0             pyhd3eb1b0_0  
    backcall                  0.2.0                    pypi_0    pypi
    blas                      1.0                         mkl  
    blinker                   1.4              py37h06a4308_0  
    bottleneck                1.3.2            py37heb32a55_1  
    brotlipy                  0.7.0           py37h27cfd23_1003  
    c-ares                    1.17.1               h27cfd23_0  
    ca-certificates           2021.7.5             h06a4308_1  
    cachetools                4.2.2              pyhd3eb1b0_0  
    certifi                   2021.5.30        py37h06a4308_0  
    cffi                      1.14.6           py37h400218f_0  
    chardet                   3.0.4           py37h06a4308_1003  
    charset-normalizer        2.0.4              pyhd3eb1b0_0  
    click                     8.0.1              pyhd3eb1b0_0  
    coverage                  5.5              py37h27cfd23_2  
    cryptography              3.4.7            py37hd23ed53_0  
    cudatoolkit               10.1.243             h6bb024c_0  
    cudnn                     7.6.5                cuda10.1_0  
    cupti                     10.1.168                      0  
    cython                    0.29.24          py37h295c915_0  
    debugpy                   1.4.1                    pypi_0    pypi
    decorator                 5.0.9                    pypi_0    pypi
    entrypoints               0.3                      pypi_0    pypi
    flatbuffers               1.12                     pypi_0    pypi
    freetype                  2.10.4               h5ab3b9f_0  
    gast                      0.3.3                    pypi_0    pypi
    google-auth               1.33.0             pyhd3eb1b0_0  
    google-auth-oauthlib      0.4.4              pyhd3eb1b0_0  
    google-pasta              0.2.0                      py_0  
    gradient-centralization-tf 0.0.3                    pypi_0    pypi
    grpcio                    1.32.0                   pypi_0    pypi
    h5py                      2.10.0           py37hd6299e0_1  
    hdf5                      1.10.6               hb1b8bf9_0  
    idna                      3.2                pyhd3eb1b0_0  
    importlib-metadata        3.10.0           py37h06a4308_0  
    intel-openmp              2021.3.0          h06a4308_3350  
    ipykernel                 6.3.1                    pypi_0    pypi
    ipython                   7.27.0                   pypi_0    pypi
    ipython-genutils          0.2.0                    pypi_0    pypi
    jedi                      0.18.0                   pypi_0    pypi
    joblib                    1.0.1              pyhd3eb1b0_0  
    jpeg                      9b                   h024ee3a_2  
    jupyter-client            7.0.2                    pypi_0    pypi
    jupyter-core              4.7.1                    pypi_0    pypi
    keras                     2.4.3                    pypi_0    pypi
    keras-preprocessing       1.1.2              pyhd3eb1b0_0  
    lcms2                     2.12                 h3be6417_0  
    ld_impl_linux-64          2.35.1               h7274673_9  
    libffi                    3.3                  he6710b0_2  
    libgcc-ng                 9.3.0               h5101ec6_17  
    libgfortran-ng            7.5.0               ha8ba4b0_17  
    libgfortran4              7.5.0               ha8ba4b0_17  
    libgomp                   9.3.0               h5101ec6_17  
    libpng                    1.6.37               hbc83047_0  
    libprotobuf               3.17.2               h4ff587b_1  
    libstdcxx-ng              9.3.0               hd4cf53a_17  
    libtiff                   4.2.0                h85742a9_0  
    libwebp-base              1.2.0                h27cfd23_0  
    lz4-c                     1.9.3                h295c915_1  
    markdown                  3.3.4            py37h06a4308_0  
    matplotlib-inline         0.1.2                    pypi_0    pypi
    mkl                       2021.3.0           h06a4308_520  
    mkl-service               2.4.0            py37h7f8727e_0  
    mkl_fft                   1.3.0            py37h42c9631_2  
    mkl_random                1.2.2            py37h51133e4_0  
    multidict                 5.1.0            py37h27cfd23_2  
    ncurses                   6.2                  he6710b0_1  
    nest-asyncio              1.5.1                    pypi_0    pypi
    numexpr                   2.7.3            py37h22e1b3c_1  
    numpy                     1.19.5                   pypi_0    pypi
    oauthlib                  3.1.1              pyhd3eb1b0_0  
    olefile                   0.46                     py37_0  
    openjpeg                  2.4.0                h3ad879b_0  
    openssl                   1.1.1l               h7f8727e_0  
    opt_einsum                3.3.0              pyhd3eb1b0_1  
    pandas                    1.3.2            py37h8c16a72_0  
    parso                     0.8.2                    pypi_0    pypi
    pexpect                   4.8.0                    pypi_0    pypi
    pickleshare               0.7.5                    pypi_0    pypi
    pillow                    8.3.1            py37h2c7a002_0  
    pip                       21.0.1           py37h06a4308_0  
    prompt-toolkit            3.0.20                   pypi_0    pypi
    protobuf                  3.17.2           py37h295c915_0  
    ptyprocess                0.7.0                    pypi_0    pypi
    pyasn1                    0.4.8                      py_0  
    pyasn1-modules            0.2.8                      py_0  
    pycparser                 2.20                       py_2  
    pygments                  2.10.0                   pypi_0    pypi
    pyjwt                     2.1.0            py37h06a4308_0  
    pyopenssl                 20.0.1             pyhd3eb1b0_1  
    pysocks                   1.7.1                    py37_1  
    python                    3.7.11               h12debd9_0  
    python-dateutil           2.8.2              pyhd3eb1b0_0  
    pytz                      2021.1             pyhd3eb1b0_0  
    pyyaml                    5.4.1                    pypi_0    pypi
    pyzmq                     22.2.1                   pypi_0    pypi
    readline                  8.1                  h27cfd23_0  
    requests                  2.26.0             pyhd3eb1b0_0  
    requests-oauthlib         1.3.0                      py_0  
    rsa                       4.7.2              pyhd3eb1b0_1  
    scikit-learn              0.24.2           py37ha9443f7_0  
    scipy                     1.6.2            py37had2a1c9_1  
    setuptools                52.0.0           py37h06a4308_0  
    six                       1.15.0                   pypi_0    pypi
    sqlite                    3.36.0               hc218d9a_0  
    tensorboard               2.4.0              pyhc547734_0  
    tensorboard-plugin-wit    1.6.0                      py_0  
    tensorflow                2.4.1           gpu_py37ha2e99fa_0  
    tensorflow-base           2.4.1           gpu_py37h29c2da4_0  
    tensorflow-estimator      2.4.0                    pypi_0    pypi
    tensorflow-gpu            2.4.1                h30adc30_0  
    termcolor                 1.1.0            py37h06a4308_1  
    threadpoolctl             2.2.0              pyhbf3da8f_0  
    tk                        8.6.10               hbc83047_0  
    tornado                   6.1                      pypi_0    pypi
    tqdm                      4.62.1             pyhd3eb1b0_1  
    traitlets                 5.1.0                    pypi_0    pypi
    typing-extensions         3.7.4.3                  pypi_0    pypi
    urllib3                   1.26.6             pyhd3eb1b0_1  
    wcwidth                   0.2.5                    pypi_0    pypi
    werkzeug                  1.0.1              pyhd3eb1b0_0  
    wheel                     0.37.0             pyhd3eb1b0_0  
    wrapt                     1.12.1           py37h7b6447c_1  
    xz                        5.2.5                h7b6447c_0  
    yarl                      1.6.3            py37h27cfd23_0  
    zipp                      3.5.0              pyhd3eb1b0_0  
    zlib                      1.2.11               h7b6447c_3  


# Reproducibility

To reproduce the solution run these python scripts in the particular order , The models are stored in models folder where as submission files are stored in submission folder

convert_data.ipynb
Train_efficientnet_b3_512.ipynb
Train_efficientnet_b5_456.ipynb
test.ipynb
