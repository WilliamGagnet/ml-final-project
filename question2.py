import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score

# read in data
df = pd.read_csv("video games sales.csv")

# adds a new column to the dataframe that distinguished a game as either a 1(a hit) or 0(not a hit)
df['Hit'] = (df['Global_Sales'] > 1.0).astype(int)

# drop unused columns from the dataframe
df = df.drop(columns=['Global_Sales', 'EU_Sales', 'JP_Sales', 'NA_Sales', 'Other_Sales', 'Rank', 'Name'])

# drop values of N/A
df = df.dropna()

# make one hot vectors from the categorical/non numeric data
categorical_features = ['Platform', 'Genre', 'Publisher']
df_encoded = pd.get_dummies(df, columns=categorical_features)

# split the data into training and testing sets; X_train, X_test, y_train, y_test
X = df_encoded.drop(columns=['Hit'])
y = df_encoded['Hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# train the logistic regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# train the decision tree model
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)


# evaluate models using accuracy, precision, recall, F1, and ROC-AUC
y_pred_lr = lr.predict(X_test)
y_pred_tree = tree.predict(X_test)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

print("Decision Tree Report:")
print(classification_report(y_test, y_pred_tree))

print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
print("Decision Tree ROC-AUC:", roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))


# features are platform, genre, publisher, and year