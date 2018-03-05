import pickle
import random
import numpy as np
import csv
import tensorflow as tf
from preprocess import (make_sparse_array, convert_to_one_dimension,
                        show_most_informative_features, get_vectorizer)
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
tf.Session(config = config)

results = {}
combined_store = []
training_number = 1000
feature_number = 251

model = Sequential()
model.add(LSTM(671, activation="relu", input_shape=(100,251)))
model.add(LSTM(1, activation="softmax"))
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with open('data/labeled.csv') as csvfile:
    data = csv.reader(csvfile)
    count = 0
    for row in data:
        if count > training_number:
            break
        else:
            label = 1 if row[2] == '1' else 0
            combined_store.append((
                                  make_sparse_array(row[1]),
                                  label))
            count += 1


random.shuffle(combined_store)


multip = round(len(combined_store) * 0.33)

train_data = combined_store[multip:]
test_data = combined_store[:multip]
print('Split test/train')
X_train = np.reshape(np.array([text for (text, label) in train_data]),
                     (len(train_data), feature_number))
y_train = np.array([label for (text, label) in train_data])
X_test = np.reshape(np.array([text for (text, label) in test_data]),
                    (len(test_data), feature_number))
y_test = np.array([label for (text, label) in test_data])
print('Fitting model')
print('X_train shape ', X_train.shape)
print('X_test ', X_test.shape)
print('y_train ', y_train.shape)
print('y_test ', y_test.shape)
print(np.array_str(X_train))
print(np.array_str(X_test))
print(np.array_str(y_test))
print(np.array_str(y_train))
'''ri = [0, 1]
for i in range(len(X_train)):
    for item in X_train[i]:
        if item not in ri:
            print('X-train ', item)
for i in range(len(X_test)):
    for item in X_test[i]:
        if item not in ri:
            print('X-test ', item)
for i in range(len(y_train)):
    if y_train[i] not in ri:
        print('y-train ', y_train[i])
for i in range(len(y_test)):
    if y_test[i] not in ri:
        print('y-test ', y_test[i])'''
model.fit(X_train, y_train, epochs=20, batch_size=100)
print('Finished fit')
score = model.evaluate(X_test, y_test, batch_size=100)
print("Score: ", score)


with open('model_sequential.pkl', 'wb') as model_writer:
    pickle.dump(model, model_writer)
