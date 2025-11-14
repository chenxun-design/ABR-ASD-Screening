"""
Main entry point for ABR-ASD Screening
"""

import argparse
from utils.data_loader import ABRDataLoader
from utils.trainer import ModelTrainer
from models.tf_tbn import TF_TBN
from models.cnn_lstm import CNN_LSTM
import config


def main():
    parser = argparse.ArgumentParser(description='ABR-ASD Screening')
    parser.add_argument('--model', choices=['tf_tbn', 'cnn_lstm', 'both'],
                       default='both', help='Model to train')
    parser.add_argument('--data_path', default='./data/processed/abr_data.csv',
                       help='Path to ABR data')
    args = parser.parse_args()

    # Load data
    data_loader = ABRDataLoader(config.Config)
    X, y = data_loader.load_data_from_csv(args.data_path)
    X_train, X_test, y_train, y_test = data_loader.preprocess_data(X, y)
    train_loader, test_loader = data_loader.create_data_loaders(X_train, X_test, y_train, y_test)

    # Train models
    if args.model in ['tf_tbn', 'both']:
        print("Training TF-TBN...")
        tf_tbn = TF_TBN(config.Config.INPUT_SIZE, config.Config.HIDDEN_SIZE,
                       config.Config.OUTPUT_SIZE, config.Config.NUM_HEADS,
                       config.Config.DROPOUT_RATE)
        tf_tbn_trainer = ModelTrainer(tf_tbn, config.Config, "TF_TBN")
        tf_tbn_trainer.train_model(train_loader, test_loader)
        results = tf_tbn_trainer.evaluate_model(test_loader)
        print(f"TF-TBN Test Accuracy: {results['accuracy']:.4f}")

    if args.model in ['cnn_lstm', 'both']:
        print("Training CNN-LSTM...")
        cnn_lstm = CNN_LSTM(config.Config.INPUT_SIZE, config.Config.HIDDEN_SIZE,
                           config.Config.OUTPUT_SIZE, config.Config.DROPOUT_RATE)
        cnn_lstm_trainer = ModelTrainer(cnn_lstm, config.Config, "CNN_LSTM")
        cnn_lstm_trainer.train_model(train_loader, test_loader)
        results = cnn_lstm_trainer.evaluate_model(test_loader)
        print(f"CNN-LSTM Test Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()