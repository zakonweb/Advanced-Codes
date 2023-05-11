"""
Download Titanic data from www.kaggle.com and do the following.
• Bring the data into appropriate form, for example, fill/remove the missing values,
encode an object column to numerical if required, etc.
• Decide the appropriate feature and target matrices.
• Draw the pair plots of all features and calculate the correlation matrix
• Based on correlation matrix, if needed, further refine the data.
• Split the data into training and test data sets.
• Fit the LinearRegression, and Gaussian Naive Bayes models to the data.
• Check the accuracy of both methods using test data.
• Compare the two methods.
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Get the directory of the currently executing script
script_dir = os.path.dirname(os.path.realpath(__file__))
#print(f"Script directory: {script_dir}")

# Create the full file path by joining the script directory and the filename
file_path = os.path.join(script_dir, 'titanic.csv')
#print(f"File path: {file_path}")

# Load the data
df = pd.read_csv(file_path)

# Rest of the code remains the same...

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df = df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

# Encode categorical data
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Draw pair plots
sns.pairplot(df)
plt.show()

# Calculate correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
# Convert to binary output for comparison
y_pred_lr = np.where(y_pred_lr > 0.5, 1, 0)
print(f"Linear Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}")

# Fit Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print(f"Gaussian Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_gnb)}")
# End of file