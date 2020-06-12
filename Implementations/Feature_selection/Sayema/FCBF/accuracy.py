from FCBF_module import FCBF
from sklearn.naive_bayes import MultinomialNB
from time import time
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
from FCBF_module import FCBF


#path_dataset = './input/mnist/'
path_dataset_save = 'C:/Users/SAYEMA/Documents/ECE657/Paper analysis/Implementation/MNIST/10000Train_5000Test/'

file = open(path_dataset_save+'X_train_picked.pckl','rb')
X_train_picked = pickle.load(file); file.close()
file = open(path_dataset_save+'X_test_picked.pckl','rb')
X_test_picked = pickle.load(file); file.close()
file = open(path_dataset_save+'y_train_picked.pckl','rb')
y_train_picked = pickle.load(file); file.close()
file = open(path_dataset_save+'y_test_picked.pckl','rb')
y_test_picked = pickle.load(file); file.close()
X_train = X_train_picked
X_test = X_test_picked
y_train = y_train_picked
y_test = y_test_picked

print('Computing embedding...')
fcbf = FCBF()
t0 = time()
fcbf.fit(X_train, y_train)

print('Done!')
print('Time: ', (time() - t0))
k = len(fcbf.idx_sel)  # Number of selected features for FCBF
print(fcbf.idx_sel)
print("Number of selected features for FCBF", k)

#Create a Gaussian Classifier
model = MultinomialNB()

# Train the model using the training sets
model.fit(X_train[:, fcbf.idx_sel], y_train)

#Predict Output
predicted = model.predict(X_test[:, fcbf.idx_sel])

acc = accuracy_score(y_test, predicted)
loss = mean_squared_error(y_test, predicted)

print("Accuracy is :", acc)
print("Loss is : ", loss)
