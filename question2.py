import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define and train logistic regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)

# train decision tree
tree = DecisionTreeClassifier(
    max_depth=15,
    class_weight='balanced',
    min_samples_split=10,
    random_state=42
)
tree.fit(X_train, y_train)

# evaluate models
y_pred_lr = lr.predict(X_test_scaled)
y_pred_tree = tree.predict(X_test)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

print("Decision Tree Report:")
print(classification_report(y_test, y_pred_tree))

print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]))
print("Decision Tree ROC-AUC:", roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))