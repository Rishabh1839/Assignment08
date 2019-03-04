# importing the necessary libraries required
from sklearn import datasets

iris = datasets.load_iris()

# We will be splitting the test data into X and Y values
# Here we will label x as our iris data
X = iris.data
# We will be labeling Y as our iris target
Y = iris.target

# Now we will be splitting this data into two partitions which will be our training and test dataa
# We use our X train/test and Y train/test for training and testing our data
from sklearn.model_selection import train_test_split

# this import splits our data  in half with the help of test_size=0.5 that splits Y and X
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=.5)

from sklearn import tree

# here we will be using our decision tree classifier
# i named my variable as decision_tree_classifier
decision_tree_classifier = tree.DecisionTreeClassifier()

# training the classifier on my training data
decision_tree_classifier.fit(trainX, trainX)

# calling out the predict method to classify the training data
predictions = decision_tree_classifier.predict(testX)

# now we will print our predictions that correspond the the type of iris predicted for each row of our data
# Lets see how accurate our classifier is on the testing data
# We compare our predicted label with true labels for calculating accuracy
from sklearn.metrics import accuracy_score

print(predictions)
print("Lets print our predictions from our decision tree classifier: ")
print(accuracy_score(testY, predictions))

from sklearn.neighbors import KNeighborsClassifier

nearest_neighbors_classifier = KNeighborsClassifier()
nearest_neighbors_classifier.fit(trainX, trainY)
prediction_from_KNeighborClassifier = nearest_neighbors_classifier.predict(testX)
print("The predictions from our KNN Classifier is: ")
print(accuracy_score(testY, prediction_from_KNeighborClassifier))
