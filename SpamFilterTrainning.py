import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

FilePath = 'spam.csv'
processedFilePath = 'Processed.csv'
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


# Extract the message and label columns
messages = data['v2'].values
labels = data['v1'].values

# Split the data into training and validation sets
train_messages, test_messages, train_labels, test_labels = train_test_split(
    messages, labels, test_size=0.2, random_state=42)

# Create a count vectorizer to convert text to a matrix of token counts
vectorizer = CountVectorizer()
train_messages_counts = vectorizer.fit_transform(train_messages)
test_messages_counts = vectorizer.transform(test_messages)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(train_messages_counts, train_labels)

# Make predictions on the test set
label_pred = clf.predict(test_messages_counts)

# Evaluate the classifier's performance
print("Accuracy:", accuracy_score(test_labels, label_pred))
print("Confusion Matrix:\n", confusion_matrix(test_labels, label_pred))
print("Classification Report:\n", classification_report(test_labels, label_pred))

with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
