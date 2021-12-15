from matplotlib.pyplot import hist
from nltk.util import pad_sequence
import preprocess as prc
import pandas as pd

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras import Sequential
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def max_len(input_str):
    return len(input_str.split(' '))

def count_words(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count

## preprocess
df = pd.read_csv (r'./spam_ham_dataset.csv')
df.drop(columns=['label', 'num'], inplace=True)

df['text'] = df['text'].apply(prc.position_tag)
df['text'] = df['text'].apply(prc.lemmatize)
df['text'] = df['text'].apply(prc.clear_text)
df['text'] = df['text'].apply(prc.tokenize)
df['text'] = df['text'].apply(prc.remove_stop_words)

max_len = 150
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.3)
words_count = count_words(df['text'])

tokenizer = Tokenizer(len(words_count))
tokenizer.fit_on_texts(X_train)

train_sequences = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding="post", truncating="post")
test_sequences = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=max_len, padding="post", truncating="post")

## network
neur_net = Sequential()

neur_net.add(Embedding(len(words_count), 32, input_length=max_len))
neur_net.add(LSTM(128, dropout=0.1))
neur_net.add(Dense(1, activation="sigmoid"))

neur_net.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

history = neur_net.fit(train_sequences, y_train, epochs=10, validation_data=(test_sequences, y_test))

train_predict = [ 1 if i > 0.5 else 0 for i in neur_net.predict(train_sequences)]
test_predict = [ 1 if i > 0.5 else 0 for i in neur_net.predict(test_sequences)]

print("Training data\n", classification_report(y_train, train_predict))
print("Test data\n", classification_report(y_test, test_predict))

matrix = confusion_matrix(y_test, test_predict)

labels = ["non-spam", "spam"]
df_cm = pd.DataFrame(matrix, index = [i for i in labels], columns = [i for i in labels])
sn.heatmap(df_cm, annot=True, fmt='.2f')
plt.show()