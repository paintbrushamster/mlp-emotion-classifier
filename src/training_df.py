# training_df

import pandas as pd
import pickle

def preprocess_data(input_training_file='twitter_training.csv', output_training_file='train_preprocessed.csv', output_test_file='test_preprocessed.csv', word_index_file='word_index.pkl'):
    # Define column names
    column_names = ['tweet_id', 'entity', 'sentiment', 'tweet_content']
    df = pd.read_csv(input_training_file, names=column_names)

    # Remove duplicates
    df = df.drop_duplicates(subset='tweet_id', keep='first')

    # Clean and lowercase text
    df['clean_tweet'] = df['tweet_content'].str.replace("[^a-zA-Z]", " ", regex=True)
    df['entity'] = df['entity'].str.replace("[^a-zA-Z]", " ", regex=True).str.lower()
    df['clean_tweet'] = df['clean_tweet'].str.lower()
    df['combined_text'] = df['entity'] + " " + df['clean_tweet']

    # Tokenization
    df['tokens'] = df['clean_tweet'].str.split()

    # Encoding sentiment labels
    sentiment_mapping = {'Irrelevant': 3, 'Positive': 2, 'Neutral': 1, 'Negative': 0}
    df['sentiment_encoded'] = df['sentiment'].map(sentiment_mapping)

    # Vectorization and padding
    max_sequence_length = 50
    unique_words = set(word for tokens in df['tokens'] for word in tokens)
    word_index = {word: i+1 for i, word in enumerate(unique_words)}

    # Save the word_index for other data processing
    with open(word_index_file, 'wb') as f:
        pickle.dump(word_index, f)


    vectorized_sequences = []
    for tokens in df['tokens']:
        vectorized_seq = [word_index.get(word, 0) for word in tokens]  # 'get' to handle unseen words
        padded_seq = vectorized_seq[:max_sequence_length] + [0]*(max_sequence_length - len(vectorized_seq))
        vectorized_sequences.append(padded_seq)
    df['vectorized_tokens'] = vectorized_sequences

    # Data splitting
    test_size = 0.1
    split_idx = int(len(df) * (1 - test_size))
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    # Save preprocessed data
    train_df.to_csv(output_training_file, index=False)
    test_df.to_csv(output_test_file, index=False)


if __name__ == "__main__":
    preprocess_data()
