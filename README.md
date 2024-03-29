# TriConvGRU
A novel time-series forecasting model applied to the Ontario Electricity Market

The proposed model in this GitHub repository is a hybridization of GRU and LSTM that is applied to the Ontario electricity market. Additionally, the paper includes a comparison of several machine learning and statistical models. The architecture of the proposed model is shown in the following:

![triconvgru (1)](https://user-images.githubusercontent.com/58978680/230785238-65a0d35c-c7b3-4eb6-923c-f4ec7d3b3ae5.png)


The repository's `.py` files comprise the implementation of various functions and operations linked to the proposed model.

1. The `DataLoader.py` contains code for converting raw data into tensors.
2. The `inference.py` has code for inference and forecasting using the test set. 
3. The `main.py`is the main file, which you should execute. 
4. The `model.py` file contains the TriconvGRU code implemented in Pytorch.
5. The `preprocessing.py` has functions for data preprocessing and time window creation. 
6. The `trainer.py` file has the train and validation functions.
7. the `utils.py` file contains functions for error metrics calculation and forecast visualization.


A sample dataset and test set are included in the repository. Also, the **main dataset** used for multivariate forecasting is attached to the repo.

To execute the model, **clone the repository** and run the command: `!python main.py`.

You can also alter the hyperparameters by running the code: `!python main.py --batch_size 32 --epoch 100 --patience 30 --num_feature 1 --for_hor 3 --n_timesteps 72`. 

To fine-tune the model, you need to define the following hyper-parameters:
1. `batch_size`: Size of each batch
2. `epoch`: Number of epochs
3. `patience`: Patience for early stopping
4. `num_feature`: Number of input features
5. `for_hor`: Forecasting horizon
6. `n_timesteps`: Number of previous timesteps

Also, in the `main.py` file of your GitHub repository, you have the option to modify other hyper-parameters for tuning. The `hyper` section is where you can make these changes. By using grid search, the program will identify the most suitable candidate for the hidden layers of GRU and CNN, as well as the optimal learning rate.

If you use the model or code in this repository, make sure to cite the corresponding paper.

Ehsani, Behdad and Pineau, Pierre-Olivier and Charlin, Laurent, Price Forecasting in the Ontario Electricity Market via TriConvGRU Hybrid Model: Univariate vs. Multivariate Frameworks (April 26, 2023). Available at SSRN: https://ssrn.com/abstract=4430405
