import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import joblib as jl

raw_data = pd.read_csv("dataset.csv")

print(raw_data.shape)
print(raw_data.head(10))
print(raw_data.info())
print(raw_data.duplicated().sum())
print(raw_data["email"].duplicated().sum())
print(raw_data["category"].value_counts(normalize=True) * 100)


raw_data["category"].value_counts().plot(kind="bar")
plt.title("Spam vs Not-Spam Email Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Emails")
plt.xticks(rotation=0)
plt.show()

data = raw_data.drop_duplicates(subset="email")

data["labels"] = data["category"].map({
  "spam" : 1,
  "not-spam" : 0
})

print(data.shape)
print(data.head())
print(data.info())
print(data.isna().sum())
print(data.duplicated().sum())
print(data["email"].duplicated().sum())
print(data["category"].value_counts(normalize=True) * 100)
print(data["labels"].value_counts())

data["labels"].value_counts().plot(kind="bar")
plt.title("0 vs 1")
plt.show()


data["length"] = data["email"].str.len()

print(data.head(10))
print(data.info())

avg_spam_len = data[data["labels"] ==1]["length"].mean()
avg_not_spam_len = data[data["labels"] == 0]["length"].mean()

print(data[data["labels"] == 1]["email"].head())
print(data[data["labels"] == 0]["email"].head())

print("Average length of spam emails: ", avg_spam_len)
print("Average length of not spam emails:", avg_not_spam_len)

X = data["email"]
y = data["labels"]

X_train, X_test, y_train, y_test = train_test_split(
  X, y,
  test_size=0.2,
  random_state=42
)

vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)

print(X_train_vectorized.shape , X_test_vectorized.shape)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)

print(accuracy)
print(confusion)
print(classification)


sns.heatmap(confusion, cmap="Blues", annot = True)
plt.show()

jl.dump(model, "model.pkl")
jl.dump(vectorizer, "vectorizer.pkl")






