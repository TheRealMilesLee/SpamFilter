import pickle

with open('spam_classifier_model.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)

with open('count_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

sample_text = ["See the details mtf Hengyi Li Dreamer | Student Developer | Content Creator Bachelor in Computer Science 2305 BNB Hall 100 E. Normal Avenue Kirksville, MO 63501 Phone: (816) 541-9947"]
sample_text_counts = loaded_vectorizer.transform(sample_text)
prediction = loaded_clf.predict(sample_text_counts)
print("Prediction:", prediction[0])
print(type(prediction[0]))
