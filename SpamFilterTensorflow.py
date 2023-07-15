import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

FilePath = 'spam.csv'
processedFilePath = 'Processed.csv'

"""
Cleanup the dataset and delete all of the non utf-8 characters. Write the processed file into Processed.csv
"""
with open(FilePath, 'rb') as csv_in:
    with open(processedFilePath, "w", encoding="utf-8") as csv_temp:
        for line in csv_in:
            if not line:
                break
            else:
                line = line.decode("utf-8", "ignore")
                csv_temp.write(str(line).rstrip() + '\n')

# Load the CSV file into a pandas dataframe
data = pd.read_csv(processedFilePath, encoding='utf-8')

# Assign label and data
messages = data['v2'].values
labels = data['v1'].values

# Split the data into training and validation sets
train_messages, test_messages, train_labels, test_labels = train_test_split(
    messages, labels, test_size=0.2, random_state=42)

# Create a count vectorizer to convert text to a matrix of token counts
vectorizer = CountVectorizer()
train_messages_counts = vectorizer.fit_transform(train_messages)
test_messages_counts = vectorizer.transform(test_messages)

# Convert the data to float32
train_messages_counts = train_messages_counts.astype('float32')
test_messages_counts = test_messages_counts.astype('float32')

# Normalize the data
train_messages_counts /= train_messages_counts.max()
test_messages_counts /= test_messages_counts.max()

# Convert labels to numerical values
train_labels = [1 if label == 'spam' else 0 for label in train_labels]
test_labels = [1 if label == 'spam' else 0 for label in test_labels]

# Convert the data to TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_messages_counts.toarray(), train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_messages_counts.toarray(), test_labels))

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(
        train_messages_counts.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_dataset.batch(512), epochs=100)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_dataset.batch(512))
print("Accuracy:", accuracy)

# Make predictions on the test set
label_pred_probs = model.predict(test_dataset.batch(512))
label_pred = np.round(label_pred_probs).flatten()

# Evaluate the model's performance
print("Confusion Matrix:\n", confusion_matrix(test_labels, label_pred))
print("Classification Report:\n", classification_report(test_labels, label_pred))

# Save the model and vectorizer to files
model.save('spam_classifier_model.h5')
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
