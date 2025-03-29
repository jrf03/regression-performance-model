# Jacob Franks
# CSCE 3201.400
# 03/28/2025

# Just took the template from the instructions document and tried to fill in the blanks
# I was a little confused on the SVM regressor creation bit because I didn't see anything in the slides about making one
    # Because of that, I got loads of errors and spent a very long time looking for solutions.
    # Hopefully the stuff I added to fix it isn't too far out of the scope of what we were intended to do.
        # If there was a simpler way of doing this successfully, please feel free to let me know. Thank you!
# I don't know if we needed to do this, but I created a requirements.txt file that shows all the packages included
# To run this, it should just be "python3 main.py" or "python main.py"

# Import the required packages
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, explained_variance_score
from sklearn.preprocessing import binarize, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Fetch dataset
student_performance = fetch_ucirepo(id=320)

# Data (as pandas dataframes)
X = student_performance.data.features
y = student_performance.data.targets

# Metadata
print(student_performance.metadata)

# Variable information
print(student_performance.variables)

# NOTE: The next three sections weren't really in the instructions, but I kept getting a ton of errors and ended up here after a lot of debugging

# Extract the final grade (G3) as target
y = y['G3'] if 'G3' in y.columns else y.iloc[:, -1]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Shuffle the data
X, y = shuffle(X, y, random_state=7)

# Split the dataset into training and testing in an 80/20 format
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Create and train the Support Vector Regressor using a linear kernal
sv_regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='linear'))
])

sv_regressor.fit(X_train, y_train)

# Run the regressor on the testing data and predict the output (predicted labels)
y_test_pred = sv_regressor.predict(X_test)

# Evaluate the performance of the regressor and print the initial metrics
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print(f"\nRegression Metrics:")
print(f"Mean squared error: {mse}")
print(f"Explained variance score: {evs}")

# Binarize the predicted values and the actual values using threshold of 12.0
y_test_reshaped = np.array(y_test).reshape(-1, 1)
y_test_pred_reshaped = y_test_pred.reshape(-1, 1)

y_test_label = binarize(y_test_reshaped, threshold=12.0).ravel()
y_pred_label = binarize(y_test_pred_reshaped, threshold=12.0).ravel()

# Create the confusion matrix using the predicted labels and the actual labels
confusion_mat = confusion_matrix(y_test_label, y_pred_label)
print("\nConfusion Matrix:")
print(confusion_mat)

# Visualize the confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()

# Print the classification report based on the confusion matrix
print("\nClassification Report:")
print(classification_report(y_test_label, y_pred_label))