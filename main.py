from DynamicDeepHit.DataPreprocessing import DataPreprocessor
from DynamicDeepHit.Trainer import Trainer
from DynamicDeepHit.Evaluation import Evaluation


if __name__ == '__main__':
    # Load the data
    print('<---Starting Model--->')
    print('Loading Data...')
    datahandler = DataPreprocessor()
    device = datahandler.device
    train_data, X_train_padded = datahandler.return_train_data()
    val_data, _ = datahandler.return_val_data()
    output_num_durations = datahandler.output_num_durations
    duration_grid_train_np = datahandler.duration_grid_train_np
    print('\nData Loaded Successfully')
    
    # Load and train the model
    print('\nTraining Model')
    trainer = Trainer(train_data, X_train_padded, val_data, output_num_durations, device)
    model = trainer.train_and_validate()

    # Evaluate
    test_data = datahandler.return_test_data()
    evaluation = Evaluation(model, test_data, duration_grid_train_np)
