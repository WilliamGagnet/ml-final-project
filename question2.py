import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# read in data into a pandas dataframe
df = pd.read_csv("video games sales.csv")

# adds a new column to the dataframe that labels each game as either a 1 (a hit) or 0 (not a hit)
df['Hit'] = (df['Global_Sales'] > 1.0).astype(int)

# Drop unused columns that are either redundant, not needed for prediction, or not helpful as features
df = df.drop(columns=['Global_Sales', 'EU_Sales', 'JP_Sales', 'NA_Sales', 'Other_Sales', 'Rank', 'Name'])

# Drop any rows that contain missing (NaN) values
df = df.dropna()

# make one hot vectors from the categorical/non numeric data
categorical_features = ['Platform', 'Genre', 'Publisher'] # get the features that were categorical
df_encoded = pd.get_dummies(df, columns=categorical_features) # makes new df with one hot vectors for the categorical features

# Split the data into features (X) and target labels (y)
X = df_encoded.drop(columns=['Hit'])
y = df_encoded['Hit']

# split the data into training and testing sets; X_train, X_test, y_train, y_test
# test_size=0.2: training set is 80% of the data, testing set is 20% of the data
# stratify=y Makes sure that the proportion of hit vs. not-hit games is the same in both training and test sets. This prevents imbalance in either set, which helps evaluation be fair and consistent.
# random_state=42 Sets the random seed so the split is reproducible. Everyone using the same seed will get the same split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model 1: logistic regression model

# creates a caler object using scikit-learn; StandardScaler standardizes the data so that Each feature (column) has a mean of 0 and a standard deviation of 1
# This helps the model converge faster and perform better, as it relies on weights
scaler = StandardScaler()

# calculates the mean and standard deviation of each feature in the training data (fit) and then transforms the data using those values (transform)
# result is a scaled version of X_train where each column is standardized
# train scaled value = (original value − mean) / standard deviation
X_train_scaled = scaler.fit_transform(X_train)

# applies the same scaling parameters to the testing set
# no fitting the scaler on the test set because you're supposed to use the same scaling rule as training
# test scaled value = (test value − training mean) / training standard deviation
# training mean and training standard deviation are both stored in the scaler object
X_test_scaled = scaler.transform(X_test)

# define the logistic regression model
# max_iter=1000 : 1000 iterations of gradient descent to find the best/optimal weights
# class_weight='balanced' Tells the model to give more importance to underrepresented classes; It automatically adjusts the weights inversely proportional to class frequencies; which means it gives more “attention” to hits so the model doesn’t ignore them
# class_weight='balanced' adjusts how much each class contributes to the loss function, so the model gives more attention to underrepresented classes during all of training.
# class weight = total number of samples / ((number of classes) * (number of samples in this class))
lr = LogisticRegression(max_iter=1000, class_weight='balanced')

# trains the defined logistic regression model
# Initializes the model weights (randomly or zero), runs gradient descent to minimize log loss and adjust weights, and repeats this for up to max_iter=1000 times (or until convergence)
# Scikit-learn stops training when The change in the loss (or weights) between iterations gets smaller than a certain threshold — called the tolerance, or tol
# By default, tol = 1e-4 which means that If the change in the loss or the gradient from one step to the next is less than 0.0001, it considers the model to have converged
# If that doesn’t happen after max_iter steps, it gives you a convergence warning
lr.fit(X_train_scaled, y_train)

# Model 2: decision tree classifier model

# define the decision tree classifier model
# max_depth=15 limits the maximum depth of the tree to 15 levels; Helps prevent overfitting
# class_weight='balanced' tells the tree to account for class imbalance; It adjusts the weight of each sample based on how rare its class is
# min_samples_split=10 means A node must have at least 10 samples to be considered for a split; Prevents the tree from splitting too aggressively on small subsets of data and helps prevent overfitting
# random_state=42 Sets the random seed for reproducibility
# While the tree is evaluating splits, each sample contributes to the information gain. When you set class_weight='balanced', the tree gives more weight to the rare class samples when calculating impurity and deciding splits.
tree = DecisionTreeClassifier(
    max_depth=15,
    class_weight='balanced',
    min_samples_split=10,
    random_state=42
)

# trains the defined decision tree classifier model
# The tree starts at the root (all data), It looks for the best feature to split the data, based on information gain, and It recursively splits the data into branches until It reaches the max_depth, Or the splits don’t improve impurity much Or nodes have too few samples
tree.fit(X_train, y_train)

# evaluate models

# Uses the trained logistic regression model to make predictions on the scaled test set; Returns a list of predicted labels (0 or 1) — one for each test sample
y_pred_lr = lr.predict(X_test_scaled)

# Same thing, but for the decision tree model; Uses the unscaled test set, since trees don’t need scaled data
y_pred_tree = tree.predict(X_test)

# These print out a full performance breakdown for each model:
#   Precision – how many predicted hits were actually hits
#   Recall – how many actual hits were caught
#   F1-score – balance between precision & recall
#   Support – number of actual samples per class
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

print("Decision Tree Report:")
print(classification_report(y_test, y_pred_tree))

# This calculates the ROC-AUC score, which measures how well the model ranks predictions from most likely to be positive (hit) to least likely.
# This gives a score between 0.5 and 1.0:
#   0.5 = random guessing
#   1.0 = perfect separation between classes
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])) # gets the probability the model assigns to class 1 (i.e., being a hit)
print("Decision Tree ROC-AUC:", roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))

'''
Precision tells you: Of all the games the model predicted as hits, how many actually were hits?
    High precision = fewer false positives
    Useful when you want to be confident in each hit prediction

Recall tells you: Of all the actual hit games, how many did the model catch?
    High recall = fewer false negatives
    Useful when missing a hit is worse than making a wrong guess

F1-score is the balance between precision and recall:
    It’s useful when you care about both and the data is imbalanced.
    A good F1 means the model is doing well overall at identifying hits

Support just tells you:
    How many actual examples of each class are in the test set


If a model has:
    Precision = 0.30
    Recall = 0.80
    F1 = 0.44
That means: The model caught 80% of real hits (good recall), but only 30% of its hit predictions were actually correct (low precision). The F1 score balances the two.

I used ROC-AUC because it shows how well the model separates the classes, regardless of the classification threshold.
'''