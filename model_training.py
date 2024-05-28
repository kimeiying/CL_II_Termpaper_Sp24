import pandas as pd

from sklearn import dummy
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

from pycantonese import segment  # For Cantonese tokenization
from matplotlib import pyplot
import seaborn

SEED = 731


def tokenization(sentence: str) -> list:
    return segment(sentence)


data = pd.read_csv("data_annotated.tsv", delimiter="\t", encoding="utf-8")

# Printing the data distribution among the four labels
print("** Distribution of the data labels **")
print(data["label"].value_counts())
print()

# Encoding the labels as integers
encoder = preprocessing.LabelEncoder()
label = encoder.fit_transform(data.label)

# Enconding the text data features
vectorizer = feature_extraction.text.CountVectorizer(
    encoding="utf-8", tokenizer=tokenization, min_df=3, max_df=0.9
)
textdata = vectorizer.fit_transform(data.text)

# Splitting the data
textdata_train, textdata_other, label_train, label_other = (
    model_selection.train_test_split(textdata, label, test_size=0.2, random_state=SEED)
)
textdata_dev, textdata_test, label_dev, label_test = model_selection.train_test_split(
    textdata_other, label_other, test_size=0.5, random_state=SEED
)

# Looping over a range of values to find the C value that maximizes the
# development accuracy
best_c = 0.0
max_dev_acc = 0.0
for C in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    logreg = linear_model.LogisticRegression(solver="liblinear", penalty="l1", C=C)
    logreg.fit(textdata_train, label_train)
    dev_acc = metrics.accuracy_score(label_dev, logreg.predict(textdata_dev))
    if dev_acc > max_dev_acc:
        max_dev_acc = dev_acc
        best_c = C
print("** Finding the best C value **")
print(f"Best C:\t{best_c}")
print(f"Max development accuracy:\t{max_dev_acc:.4f}")

# Using `best_c` to fit the model again
logreg = linear_model.LogisticRegression(solver="liblinear", penalty="l1", C=best_c)
logreg.fit(textdata_train, label_train)
test_acc = metrics.accuracy_score(label_test, logreg.predict(textdata_test))
print(f"Testing accuracy:\t{test_acc:.4f}")
print()

# Comparing to a baseline model
dummy_model = dummy.DummyClassifier()
dummy_model.fit(textdata_train, label_train)
dummy_dev_acc = metrics.accuracy_score(label_dev, dummy_model.predict(textdata_dev))
dummy_test_acc = metrics.accuracy_score(label_test, dummy_model.predict(textdata_test))
print("** Comparing to a baseline model **")
print(f"Development baseline accuracy:\t{dummy_dev_acc:.4f}")
print(f"Testing baseline accuracy:\t{dummy_test_acc:.4f}")
print()

# Generating a heatmap
label_pred = logreg.predict(textdata_test)
confusion_matrix = metrics.confusion_matrix(label_test, label_pred)
category_labels = ["cantonese", "mixed", "neutral", "SWC"]

pyplot.figure(figsize=(7, 5))
seaborn.heatmap(
    confusion_matrix,
    annot=True,
    cmap="Blues",
    xticklabels=category_labels,
    yticklabels=category_labels,
)
pyplot.xlabel("Predicted Labels")
pyplot.ylabel("True Labels")
pyplot.show()
