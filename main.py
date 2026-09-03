
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
# Data loading
df=pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
print(df.head())
print(df.describe())

#Data Seperation (X axis,Y axis) & train/test split


X_raw = df.drop("Diabetes_012",axis=1).values #contains the features, not the diagnosis
Y_1=df["Diabetes_012"].values#contains only the diagnosis

#Split :  train(80%) και test (20%)

X_train_raw,X_test_raw,y_train_1,y_test_1 = train_test_split(X_raw,Y_1,test_size=0.2,random_state=42)

# X_train_raw : 80% of X_raw. The features used to train the neural network. Not normalized yet(raw).
# X_test_raw  : 20% of X_raw. The unseen features kept aside to test/evaluate the network later ("raw").
# y_train_1   : 80% of Y_1. The actual diagnoses (0, 1, or 2) matching the training features.
# y_test_1    : 20% of Y_1. The actual diagnoses (0, 1, or 2) matching the test features.

#Calculate mean and standard deviation from the training set only (to prevent data leakage)
X_mean = np.mean(X_train_raw,axis=0)
X_Std=np.std(X_train_raw,axis=0)

#Normalize  train and test set
X_train = (X_train_raw-X_mean)/X_Std
X_test=(X_test_raw-X_mean)/X_Std

#one-hot encoding

y_train=np.eye(3)[y_train_1.astype(int)]
y_test=np.eye(3)[y_test_1.astype(int)]

#Dimensions : features * patients (That is why we take the .T)
train_x = X_train.T
train_y = y_train.T
test_x = X_test.T
test_y = y_test.T

def softmax(Z):
    exp_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exp_z / np.sum(exp_z, axis=0, keepdims=True)
    cache = Z
    return A, cache


def ReLU(z) :
   A = np.maximum(0,z)
   cache = z
   return A,cache


def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA,copy=True)
    dZ[Z <= 0] = 0
    return  dZ


def initialize_parameters(n_x, n_h1, n_h2, n_h3, n_y):
    np.random.seed(42)
    W1 = np.random.randn(n_h1, n_x) * np.sqrt(2/n_x)
    b1 = np.zeros((n_h1, 1))

    W2 = np.random.randn(n_h2, n_h1) * np.sqrt(2/n_h1)
    b2 = np.zeros((n_h2, 1))

    W3 = np.random.randn(n_h3, n_h2) * np.sqrt(1/n_h2)
    b3 = np.zeros((n_h3, 1))

    W4 = np.random.randn(n_y, n_h3) * np.sqrt(2/n_h3)
    b4 = np.zeros((n_y, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3 , "W4" :W4 , "b4": b4}

def linear_forward(A_prev,W,b):
    Z = np.dot(W,A_prev) + b
    cache = (A_prev,W,b)
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    Z,linear_cache = linear_forward(A_prev,W,b)
    if activation=="relu":
        A,activation_cache = ReLU(Z)
    elif activation=="softmax":
        A,activation_cache = softmax(Z)

    cache = (linear_cache,activation_cache)
    return A,cache


def compute_cost(AL,Y):
    m=Y.shape[1]
    cost = -(1 / m) * np.sum(Y * np.log(AL + 1e-9))
    cost = float(np.squeeze(cost))
    return cost



def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m=A_prev.shape[1]
    dW=(1/m) * np.dot(dZ,A_prev.T)
    db=(1/m)*np.sum(dZ,axis=1,keepdims = True)
    dA_prev = np.dot(W.T,dZ)

    return dA_prev,dW,db

def linear_activation_backward(dA_or_dZ,cache,activation):
    linear_cache,activation_cache = cache
    if activation == "relu":
        dZ= relu_backward(dA_or_dZ,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    elif activation=="softmax":
        dZ = dA_or_dZ
        dA_prev,dW,db = linear_backward(dZ,linear_cache)

    return dA_prev,dW,db

#
# def update_parameters(parameters, grads, learning_rate):
#
#     parameters["W1"] -= learning_rate * grads["dW1"]
#     parameters["b1"] -= learning_rate * grads["db1"]
#
#
#     parameters["W2"] -= learning_rate * grads["dW2"]
#     parameters["b2"] -= learning_rate * grads["db2"]
#
#
#     parameters["W3"] -= learning_rate * grads["dW3"]
#     parameters["b3"] -= learning_rate * grads["db3"]
#
#     return parameters


def random_mini_batches(X,Y,mini_batch_size):
    m=X.shape[1]
    mini_batches =[]

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]
    total_mini_batches = math.floor(m/mini_batch_size)

    for i in range(total_mini_batches):
        mini_batch_X = shuffled_X[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_Y=shuffled_Y[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batches.append((mini_batch_X,mini_batch_Y))



    if m%total_mini_batches!=0:
        mini_batch_X = shuffled_X[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


def initialize_adam(parameters):
    L = len(parameters)//2
    v = {}
    s = {}
    for i in range (1,L+1):
        v["dW" + str(i)] = np.zeros_like(parameters["W" + str(i)])
        v["db" + str(i)] = np.zeros_like(parameters["b" + str(i)])
        s["dW" + str(i)] = np.zeros_like(parameters["W" + str(i)])
        s["db" + str(i)] = np.zeros_like(parameters["b" + str(i)])

    return v,s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        # Momentum (v)
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        # RMSprop (s)
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])

        # Bias correction
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))


        #update with Adam
        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (
                    np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (
                    np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s
def three_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=True):
    initial_learning_rate = learning_rate
    decay_rate = 0.01
    np.random.seed(42)
    grads = {}
    costs = []
    (n_x, n_h1,n_h2, n_h3,n_y) = layers_dims
    parameters = initialize_parameters(n_x,n_h1,n_h2,n_h3,n_y)
    v,s = initialize_adam(parameters)
    t=0

    for i in range(num_iterations):
        learning_rate = initial_learning_rate / (1 + decay_rate * i)
        minibatches = random_mini_batches(X,Y,mini_batch_size=128)
        total_cost = 0

        for minibatch in minibatches:
            (minibatch_X,minibatch_Y) = minibatch
            t+=1


            A1, cache1 = linear_activation_forward(minibatch_X, parameters["W1"], parameters["b1"], "relu")


            A2, cache2 = linear_activation_forward(A1, parameters["W2"], parameters["b2"], "relu")

            A3, cache3 = linear_activation_forward(A2, parameters["W3"], parameters["b3"], "relu")


            AL, cache4 = linear_activation_forward(A3, parameters["W4"], parameters["b4"], "softmax")

            total_cost += compute_cost(AL,minibatch_Y)

            dZL = AL - minibatch_Y
            dA3, dW4, db4 = linear_activation_backward(dZL, cache4, "softmax")
            dA2, dW3, db3 = linear_activation_backward(dA3, cache3, "relu")
            dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "relu")
            _, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

            grads["dW1"] = dW1
            grads["db1"] = db1
            grads["dW2"] = dW2
            grads["db2"] = db2
            grads["dW3"] = dW3
            grads["db3"] = db3
            grads["dW4"] = dW4
            grads["db4"] = db4

            parameters ,v,s= update_parameters_with_adam(parameters, grads, v, s, t, learning_rate)
        epoch_cost = total_cost/len(minibatches)

        if print_cost and i%100 == 0:
            print(f"Epoch {i} | Loss : {epoch_cost:.4f}")
            costs.append(epoch_cost)
    return parameters,costs


def predict(X, parameters):

    A1, _ = linear_activation_forward(X, parameters["W1"], parameters["b1"], "relu")

    A2, _ = linear_activation_forward(A1, parameters["W2"], parameters["b2"], "relu")

    A3,_ = linear_activation_forward(A2, parameters["W3"], parameters["b3"], "relu")

    probs, _ = linear_activation_forward(A3, parameters["W4"], parameters["b4"], "softmax")
    return probs



layers_dims = (21, 64, 32, 16 , 3)
parameters, costs = three_layer_model(train_x, train_y, layers_dims=layers_dims, learning_rate=0.001, num_iterations=800)
# Loss Curve visualization

plt.figure(figsize=(8, 5))
plt.plot(np.squeeze(costs), color='#1f77b4', linewidth=2, marker='o')
plt.ylabel('Cost (Loss)', fontsize=12)
plt.xlabel('Iterations (x100 epochs)', fontsize=12)
plt.title('Neural Network Learning Curve\n(Adam Optimizer, LR=0.001)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
print("\n")
#  Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train_1)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Κάνουμε προβλέψεις με τον Naive Bayes στο Test Set
y_pred_nb = nb_model.predict(X_test)

# 2. Υπολογίζουμε τον πίνακα σύγχυσης
cm = confusion_matrix(y_test_1, y_pred_nb)

# 3. Σχεδιάζουμε και αποθηκεύουμε το διάγραμμα
plt.figure(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Prediabetes', 'Diabetes'])
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
plt.title('Naive Bayes - Confusion Matrix', fontsize=14, fontweight='bold')
plt.savefig("naive_bayes_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("\n")
plt.close()



print("Welcome to diabetes analysis and prediction : \n")
print("Before we start,you must answer to 21 quick questions about you!!")
questions = [
    ("1. Do you have high blood pressure? (0 = No, 1 = Yes): ", 0, 1),
    ("2. Do you have high cholesterol? (0 = No, 1 = Yes): ", 0, 1),
    ("3. Have you had a cholesterol check in the past 5 years? (0 = No, 1 = Yes): ", 0, 1),
    ("4. What is your Body Mass Index (BMI)? (e.g., 25): ", 10, 99),
    ("5. Have you smoked more than 100 cigarettes in your life? (0 = No, 1 = Yes): ", 0, 1),
    ("6. Have you ever had a stroke? (0 = No, 1 = Yes): ", 0, 1),
    ("7. Do you have coronary heart disease or a history of heart attack? (0 = No, 1 = Yes): ", 0, 1),
    ("8. Have you done any physical activity or exercise in the past 30 days? (0 = No, 1 = Yes): ", 0, 1),
    ("9. Do you eat at least one fruit per day? (0 = No, 1 = Yes): ", 0, 1),
    ("10. Do you eat vegetables at least once per day? (0 = No, 1 = Yes): ", 0, 1),
    ("11. Do you consume alcohol heavily? (0 = No, 1 = Yes): ", 0, 1),
    ("12. Do you have health insurance? (0 = No, 1 = Yes): ", 0, 1),
    ("13. Have you avoided visiting a doctor due to cost in the past year? (0 = No, 1 = Yes): ", 0, 1),
    ("14. How would you rate your general health? (1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor): ", 1, 5),
    ("15. How many days in the past month did you experience poor mental health? (0–30): ", 0, 30),
    ("16. How many days in the past month did you experience poor physical health? (0–30): ", 0, 30),
    ("17. Do you have difficulty walking or climbing stairs? (0 = No, 1 = Yes): ", 0, 1),
    ("18. Sex (0 = Female, 1 = Male): ", 0, 1),
    ("19. Age category (Scale 1–13, e.g., 1 = 18–24, 5 = 40–44, 9 = 60–64, 13 = 80+): ", 1, 13),
    ("20. Education level (Scale 1–6, e.g., 6 = College/University degree): ", 1, 6),
    ("21. Income category (Scale 1–8, e.g., 8 = Above $75k): ", 1, 8)
]
print("\n")
user_answers=[]
for text,min_val,max_val in questions:
    while True:
        user_input=input(text)
        try:
            value = float(user_input)
            if min_val<= value <= max_val:
                user_answers.append(value)
                break
            else:
                print(f"\nWrong input,your answer must be between {min_val} and {max_val}")
        except ValueError:
            print("\nPlease give a valid number,not letters")


user_array = np.array(user_answers).reshape(1,21)
user_normalized = (user_array - X_mean) / X_Std

#NN prediction
prediction = predict(user_normalized.T, parameters)
print("\n" + "*" *40)
print(" FINAL NETWORK DIAGNOSIS ")
print("*" *40)
print(f"Probability of being Healthy:    {prediction[0][0] * 100:.2f}%")
print(f"Probability of Prediabetes:  {prediction[1][0] * 100:.2f}%")
print(f"Probability of Diabetes:       {prediction[2][0] * 100:.2f}%")
print("*" *50)

# Naive Bayes Prediction
nb_prediction = nb_model.predict(user_normalized)
category_map = {0: "Healthy", 1: "Prediabetes", 2: "Diabetes"}
print("*" * 40)
print(" Naive Bayes Prediction ")
print(f"Naive Bayes predicts category: {int(nb_prediction[0])} ({category_map[int(nb_prediction[0])]})")
print("*" * 40)