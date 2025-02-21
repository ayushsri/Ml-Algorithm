# k-Nearest Neighbors (k-NN) in Python

k-Nearest Neighbors (k-NN) is a simple, yet powerful supervised machine learning algorithm used for classification and regression tasks. It works by finding the `k` closest data points (neighbors) to a given query point and making predictions based on the majority class (for classification) or average value (for regression) of these neighbors.

## Requirements

Before we start, make sure you have the following libraries installed:
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (optional, for visualization)

You can install these libraries using pip:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Steps to Implement k-NN

1. **Import Libraries**
2. **Load Dataset**
3. **Preprocess Data**
4. **Split Data into Training and Testing Sets**
5. **Train the k-NN Model**
6. **Make Predictions**
7. **Evaluate the Model**
8. **Visualize Results (Optional)**

### 1. Import Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
```

### 2. Load Dataset

For this example, we'll use the famous Iris dataset. You can load it directly from scikit-learn.

```python
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for easier manipulation (optional)
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print(df.head())
```

### 3. Preprocess Data

Standardize the features to have mean 0 and variance 1.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 4. Split Data into Training and Testing Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### 5. Train the k-NN Model

```python
# Initialize the k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)
```

### 6. Make Predictions

```python
# Make predictions on the test set
y_pred = knn.predict(X_test)
```

### 7. Evaluate the Model

```python
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
```

### 8. Visualize Results (Optional)

Here, we'll visualize the decision boundaries of the k-NN classifier. Note that this is only feasible for 2D data, so we'll use only the first two features of the Iris dataset for visualization.

```python
# Plot decision boundaries
def plot_decision_boundaries(X, y, model, title):
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # We only take the first two features for visualization purposes
    X = X[:, :2]

    # Train the model
    model.fit(X, y)

    # Plot the decision boundary by assigning a color in the color map
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()

# Visualize decision boundaries
plot_decision_boundaries(X_train, y_train, knn, title="3-Class classification (k=3)")
```

## Conclusion

You've successfully implemented the k-Nearest Neighbors (k-NN) algorithm in Python using the scikit-learn library. You learned how to preprocess data, train the model, make predictions, and evaluate the model's performance. Additionally, you visualized the decision boundaries of the k-NN classifier.

Feel free to experiment with different values of `k` and other datasets to see how the k-NN algorithm performs in different scenarios.
