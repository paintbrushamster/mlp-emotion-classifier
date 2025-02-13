# validation_df

import pandas as pd
import pickle

def preprocess_validation_data(input_file='twitter_validation.csv', output_file='validation_preprocessed.csv', word_index_file='word_index.pkl'):
    # Load word_index created during training preprocessing
    with open(word_index_file, 'rb') as f:
        word_index = pickle.load(f)
    
    
    # Define column names
    column_names = ['tweet_id', 'entity', 'sentiment', 'tweet_content']
    df = pd.read_csv(input_file, names=column_names)

    # Remove duplicates
    df = df.drop_duplicates(subset='tweet_id', keep='first')

    # Clean and lowercase text
    df['clean_tweet'] = df['tweet_content'].str.replace("[^a-zA-Z]", " ", regex=True)
    df['clean_tweet'] = df['clean_tweet'].str.lower()
    df['entity'] = df['entity'].str.replace("[^a-zA-Z]", " ", regex=True).str.lower()
    df['combined_text'] = df['entity'] + " " + df['clean_tweet']  

    # Tokenisation
    df['tokens'] = df['clean_tweet'].str.split()

    # Encoding sentiment labels
    sentiment_mapping = {'Irrelevant': 3,'Positive': 2, 'Neutral': 1, 'Negative': 0}
    df['sentiment_encoded'] = df['sentiment'].map(sentiment_mapping)

    # Vectorisation and padding
    max_sequence_length = 50
    
    vectorized_sequences = []
    for tokens in df['tokens']:
        # Handle unseen words
        vectorized_seq = [word_index.get(word, 0) for word in tokens]
        padded_seq = vectorized_seq[:max_sequence_length] + [0]*(max_sequence_length - len(vectorized_seq))
        vectorized_sequences.append(padded_seq)
    df['vectorized_tokens'] = vectorized_sequences

    # Save preprocessed data
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess_validation_data()
