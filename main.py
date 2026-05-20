import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# READ DATASET

data = pd.read_csv("news.csv")

print("===== DATASET =====")
print(data)

# INPUT AND OUTPUT

X = data['text']

y = data['label']

# CONVERT TEXT TO NUMBERS

vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)

# SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)

# CREATE MODEL

model = LogisticRegression()

# TRAIN MODEL

model.fit(X_train, y_train)

# PREDICTION

prediction = model.predict(X_test)

print("\n===== PREDICTIONS =====")
print(prediction)

# ACCURACY

accuracy = accuracy_score(y_test, prediction)

print("\n===== ACCURACY =====")
print(accuracy * 100)

# USER INPUT

print("\n===== FAKE NEWS DETECTION =====")

news = input("Enter News Text: ")

news_vector = vectorizer.transform([news])

result = model.predict(news_vector)

print("\nFINAL RESULT =", result[0])