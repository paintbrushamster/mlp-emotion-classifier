import pickle
from training_df import preprocess_data
from validation_df import preprocess_validation_data
from training_model import train_or_load_model
from validation import validate_model, load_validation_data

def main():
    # File paths
    input_training_file = 'twitter_training.csv'
    output_training_file = 'train_preprocessed.csv' 
    
    input_validation_file = 'twitter_validation.csv'
    output_validation_file = 'validation_preprocessed.csv'

    output_test_file = 'test_preprocessed.csv' # for testing, didn't implement

    word_index_file = 'word_index.pkl'

    model_filepath = "model_parameters.pkl"

     # Preprocess data
    preprocess_data(input_training_file, output_training_file, output_test_file, word_index_file)
    preprocess_validation_data(input_validation_file, output_validation_file, word_index_file)


    # Train the model or load if already exists
    model_parameters = train_or_load_model(output_training_file, model_filepath)

    # Load validation data
    X_val, y_val = load_validation_data(output_validation_file)

    # Validate the model
    validate_model(model_parameters, X_val, y_val)

if __name__ == "__main__":
    main()