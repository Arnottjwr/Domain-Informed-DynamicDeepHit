import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from .utils import discretize, _get_padded_features, zip_features

class DataPreprocessor:

    def __init__(self):
        with open('config.yaml') as yaml_data:
            self.config = yaml.safe_load(yaml_data)
        self.data_config = self.config['load_data']
        self.df = pd.read_csv(self.data_config['path'])
        X, Y, D = self.sort_data()
        self.split_data(X, Y, D)
        self.define_discrete_grid()
        self.device = torch.device(self.config['misc']['cuda'] if torch.cuda.is_available() else 'cpu')

    def sort_data(self):
        """Sort Data into Observations, Times and Outcomes"""
        self.df['gender'] = np.where(self.df['gender']=='F',1,0) # female = 1, male = 0
        features = self.df[self.data_config['features_before_preprocessing']].to_numpy()#.astype('float32')
        observed_times = (self.df['years'] - self.df['year']).to_numpy().astype('float32') # Calculate Time to Events
        event_indicators = self.df['event'].map({event:i+1 for i, event in enumerate(self.config['events'])}).fillna(0).astype('int32').to_numpy() ##TODO - Test this
        X, Y, D = [], [], []
        for id in sorted(list(set(self.df['subject_id']))):
            mask = (self.df['subject_id'] == id)
            X.append(features[mask])
            Y.append(observed_times[mask])
            D.append(event_indicators[mask])

        return X, Y, D
    
    def split_data(self, X, Y, D):
        """Split the data into training, testing and validation sets"""
        X_full_train_raw_list, self.X_test_list, Y_full_train_list, self.Y_test_list, D_full_train_list, self.D_test_list\
              = train_test_split(X, Y, D, test_size=.3, random_state=0) ##TODO - Remove hard-coded random state

        self.X_train_list, self.X_val_list, self.Y_train_list, self.Y_val_list, self.D_train_list, self.D_val_list\
              = train_test_split(X_full_train_raw_list, Y_full_train_list, D_full_train_list, test_size=.2, random_state=0)


    def define_discrete_grid(self):
        """Define the discrete grid for training to occur on"""
        self.Y_train_discrete_np, self.duration_grid_train_np = discretize(self.Y_train_list, self.config["num_durations"])
        self.Y_val_discrete_np, _ = discretize(self.Y_val_list, len(self.duration_grid_train_np) - 1, self.duration_grid_train_np)
        self.output_num_durations = len(self.duration_grid_train_np)


    def return_train_data(self):
        """Return training data"""
        train_data, X_train_padded = zip_features(self.X_train_list, self.Y_train_discrete_np, self.D_train_list,device=self.device)
        return train_data, X_train_padded 
    

    def return_val_data(self):
        """Return Validation data"""
        val_data, X_val_padded = zip_features(self.X_val_list, self.Y_val_discrete_np, self.D_val_list,device=self.device)
        return val_data, X_val_padded


    def return_test_data(self):
        X_test_padded = torch.from_numpy(_get_padded_features(self.X_test_list)).type(torch.float32).to(self.device)
        Y_train_np = np.array([_[-1] for _ in self.Y_train_list])
        D_train_np = np.array([_[-1] for _ in self.D_train_list])
        Y_test_np = np.array([_[-1] for _ in self.Y_test_list])
        D_test_np = np.array([_[-1] for _ in self.D_test_list])
        return X_test_padded, Y_train_np, D_train_np, Y_test_np, D_test_np