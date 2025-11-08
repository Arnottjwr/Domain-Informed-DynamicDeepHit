import json
import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch


from utils import load_data, compute_brier_competing, discretize, _get_padded_features, zip_features


class DomainInformedModel:

    def __init__(self):
        with open('config.yaml') as yaml_data:
            self.config = yaml.safe_load(yaml_data)

        np.random.seed(self.config['misc']['seed'])
        self.events = self.config['events']
        self.device = torch.device(self.config['misc']['cuda'] if torch.cuda.is_available() else 'cpu')


    """
    - initialise model params 
    - run training loop 
    - run evaluation metrics
    - store config and evaluation metrics
    """
    
    def load_data(self):
        """Load the dataset and preproccess it"""
        local_config = self.config['load_data']
        df = pd.read_csv(local_config['path'])
        
        features = df[local_config['features_before_preprocessing']].to_numpy().astype('float32')    
        observed_times = (df['years'] - df['year']).to_numpy().astype('float32') # Calculate Time to Events
        event_indicators = df['event'].map({event:i+1 for i, event in enumerate(self.events)}).fillna(0).astype('int32').to_numpy() ##TODO - Test this
        X = []
        Y = []
        D = []
        for id in sorted(list(set(df['subject_id']))):
            mask = (df['subject_id'] == id)
            X.append(features[mask])
            Y.append(observed_times[mask])
            D.append(event_indicators[mask])

        X_full_train_raw_list, self.X_test_list, Y_full_train_list, self.Y_test_list, D_full_train_list, self.D_test_list = train_test_split(X, Y, D, test_size=.3, random_state=0)
        X_train_list, X_val_list, self.Y_train_list, Y_val_list, self.D_train_list, D_val_list = train_test_split(X_full_train_raw_list, Y_full_train_list, D_full_train_list, test_size=.2, random_state=0)

        Y_train_discrete_np, self.duration_grid_train_np = discretize(self.Y_train_list, {self.config["num_durations"]})
        Y_val_discrete_np, _ = discretize(Y_val_list, len(self.duration_grid_train_np) - 1, self.duration_grid_train_np)
        self.output_num_durations = len(self.duration_grid_train_np)

        self.train_data, self.X_train_padded = zip_features(X_train_list, Y_train_discrete_np, self.D_train_list,device=self.device)
        self.val_data, _ = zip_features(X_val_list, Y_val_discrete_np, D_val_list,device=self.device)



    def init_model(self):
        pass

    def train_and_validate(self):
        pass

    def run_evaluation_metrics(self):
        pass

    def store_results(self):
        pass




    def main(self):
        """Main call function"""
        self.load_data()
