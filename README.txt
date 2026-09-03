                         This project is for educational purposes only and is not intended for medical diagnosis.





1) General info about the project:
This project is a custom-built Deep Neural Network in Python that predicts a person's risk of developing diabetes.
It is based on real data from a telephone survey where almost 253,680 people were asked 21 questions about their
general health and lifestyle. The data was obtained from the 2015 Behavioral Risk Factor Surveillance System (BRFSS),
which is conducted annually by the U.S. Centers for Disease Control and Prevention (CDC). The model processes answers to
these 21 questions to calculate the probability of them being Healthy, having Pre-diabetes, or having Diabetes.


                                   Understanding the Results & Data Imbalance:
In the general population, the vast majority of people are healthy. In this specific dataset, about 84% of the participants
are healthy, around 14% have diabetes, and only 2% have pre-diabetes. Because the neural network is trained on such highly
imbalanced data, it naturally develops a bias towards predicting that someone is healthy. As a result, the network might
output seemingly small percentage probabilities for pre-diabetes and diabetes, even if the user provides poor health indicators.
However, the most accurate way to interpret these results is to compare them as a multiple of the average person's probability (Relative Risk).
For example, if the model calculates a 10% probability of pre-diabetes for a user, we perform the calculation 10% ÷ 2%
(where 2% represents the actual baseline rate from the CDC study). This means the user is actually 5 times more likely
to develop the condition than the average person!

2) Architecture and algorithms

Deep neural network , with 3 hidden layers. The input is 21 features,followed by the first layer(64 neurons) ,
the second hidden(32 neurons) and the last hidden (16 neurons).
The output is a three neuron decision , which provides the probabilities for: Healthy, Pre-diabetes, and Diabetes.

the optimization techniques that used, in order to make out model as accurate as possible.
i)Mini-batch gradient descent
ii)Adam optimizer
iii)Learning rate Decay


3) Libraries
matplotlib( for loss visualization)
numpy(for mathematical operations,matrix multiplications and backprop)
pandas(loading and manage the csv dataset)
scikit-learn (Naive Bayes / data split)


4)  How to run
Make sure you have downloaded the diabetes_012_health_indicators_BRFSS2015.csv dataset and placed it in the same directory as the script.
Run the script.It will take around TEN MINUTES  ,depending on your CPU. It is totally normal due to the depth of the network(hidden layers) ,
the number of neurons , and the heavy mathematical calculations of the Adam optimizer running in pure numpy.
After that you will answer 21 questions and you will get your ratings!!!


5) Conclusion

The neural network was trained from scratch for 800 epochs using Mini-batch Gradient Descent (batches of 128) and the Adam Optimization algorithm.
During training, the Loss started at 0.4181 and successfully converged to 0.3650.

In medical datasets based on self-reported questionnaires, achieving a loss close to 0.00 is not only impossible but also undesirable,
as it would indicate severe Overfitting (memorizing the training data). Because human health involves inherent randomness and
unmeasured variables (Bayes Error), two individuals with the exact same 21 answers might have different actual diagnoses.
The drop to 0.3844 proves that the multi-layer architecture successfully bypassed the "easy" route of constantly predicting
the majority class (Healthy), and managed to extract deeper, meaningful underlying medical patterns without memorizing the dataset.


