from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import preprocess as prc
import seaborn as sn
import pandas as pd

df = pd.read_csv (r'./spam_ham_dataset.csv')
df.drop(columns=['label', 'num'], inplace=True)

df['text'] = df['text'].apply(prc.position_tag)
df['text'] = df['text'].apply(prc.lemmatize)
df['text'] = df['text'].apply(prc.clear_text)
df['text'] = df['text'].apply(prc.tokenize)
df['text'] = df['text'].apply(prc.remove_stop_words)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.3)

tfid_vectorizer = TfidfVectorizer()
model = ComplementNB()

model.fit(tfid_vectorizer.fit_transform(X_train), y_train)

x_train_predict = model.predict(tfid_vectorizer.transform(X_train))


print("Training data\n", classification_report(y_train, model.predict(tfid_vectorizer.transform(X_train))))
print("Test data\n", classification_report(y_test, model.predict(tfid_vectorizer.transform(X_test))))

matrix = confusion_matrix(y_test, model.predict(tfid_vectorizer.transform(X_test)))

labels = ["non-spam", "spam"]
df_cm = pd.DataFrame(matrix, index = [i for i in labels], columns = [i for i in labels])
sn.heatmap(df_cm, annot=True, fmt='.2f')
plt.show()