import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def prepare_data_for_training():

    #importing that will be used
    seeds_dataset = pd.read_csv("./data/seeds_dataset.csv")

    #verification of dataset
    print("\n\n---------------- Verification of the used dataset ----------------\n\n")
    seeds_dataset.info()
    print(seeds_dataset.head())
    print("\n\n----------------------------------------------------------------\n\n")

    #separation of the features and labels 
    seeds_features = seeds_dataset.loc[:, seeds_dataset.columns != 'Class(1,2,3)'] 
    labels = seeds_dataset['Class(1,2,3)']

    #scaling features
    min_max_scaler = MinMaxScaler()
    scaled_seeds_dataset = min_max_scaler.fit_transform(seeds_features)

    #partitionning dataset for a training phase and testing phase
    return train_test_split(seeds_features, labels, test_size=0.25)
