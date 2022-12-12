# ShopperIntentPredictionClickstream
Using session clickstream data to predict shopper intent.

DataCleaning.ipynb takes the data found from [1] and it is briefly examined. New features are made. The data is transformed to be handled by the ANN's. It is then saved to a different file. 

ModelDefinitions.py contains the class definitions of the RNN, LSTM, and GRU.

TrainFunctions.py contains the training function and the function used to train.

earlstopping.py is a function for early stopping.

main.ipynb uses all these functions to train and test a model

metrics.py is a file creating all the functions used to evaluate the model and model performance.


Data can be found at:
[1] Requena, B., Cassani, G., Tagliabue, J., Greco, C., and Lacasa, L. Shopper intent prediction from clickstream e-commerce data with minimal browsing information. Scientific Reports, 10(1):16983, Oct 2020. ISSN 2045- 2322. doi: 10.1038/s41598-020-73622-y. URL https://doi.org/10.1038/s41598-020-73622-y.
