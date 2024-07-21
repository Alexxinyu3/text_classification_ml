"""
Word Embeddings - Word2vec
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# load the data
file_path = 'D:/Desktop/study in France/ESIGELEC-study/Intership/IPSOS/cleaned_data_for_model.xlsx'
# file_path = 'D:/Desktop/study in France/ESIGELEC-study/Intership/IPSOS/cleaned_Prediction_file.xlsx'   # this is for data prediction
df = pd.read_excel(file_path).dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Extracting cleaned_text columns from a DataFrame
documents = df['cleaned_text']


# Preprocessing - Tokenize the text
def preprocess_text(text):
    return word_tokenize(text.lower())

df['tokenized_text'] = df['cleaned_text'].apply(preprocess_text)

# View tokenized data
print(df['tokenized_text'].head())

# Train Word2Vec model
sentences = df['tokenized_text'].tolist()  # Converting a DataFrame to a list of sentences
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# View the vector for a specific word
word = 'dasani'
if word in model.wv:
    print(f"Vector for '{word}': {model.wv[word]}")
else:
    print(f"Word '{word}' not in vocabulary.")

# View similarity between two words
word1 = 'dasani'
word2 = 'water'
if word1 in model.wv and word2 in model.wv:
    similarity = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity}")
else:
    print(f"One of the words '{word1}' or '{word2}' not in vocabulary.")

# Find the most similar words to a specific word
most_similar = model.wv.most_similar('dasani', topn=5)
print(f"Words most similar to 'dasani': {most_similar}")

# Save the model
model.save("word2vec.model")

# # Load the model
# model = Word2Vec.load("word2vec.model")

texts = df['Sound Bite Text_clean'].apply(lambda x: x.split()).tolist()
labels = df['sentiment'].tolist()

# Convert text into feature vectors
def text_to_vector(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    vector = np.mean(vectors, axis=0)
    return vector

# Creating the Feature Matrix
features = np.array([text_to_vector(text, model) for text in texts])

# Organize the data into DataFrame
df = pd.DataFrame(features)
# df.insert(0, 'label', labels)

# save
# df.to_csv('data/data_for_model.csv', index=False)
df.to_csv('data/data_for_model.csv', index=False)
# df.to_csv('data/data_for_preds.csv', index=False)

print("Data saved to output.csv")