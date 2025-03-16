from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data  
y = iris.target  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy of the kNN model:", accuracy_knn)

weighted_knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance')

weighted_knn_model.fit(X_train, y_train)

y_pred_weighted_knn = weighted_knn_model.predict(X_test)

accuracy_weighted_knn = accuracy_score(y_test, y_pred_weighted_knn)
print("Accuracy of the kNN model with weighting:", accuracy_weighted_knn)

k_values = list(range(1, 21))
mean_accuracies = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    accuracies = cross_val_score(knn_model, X, y, cv=5)  
    mean_accuracies.append(np.mean(accuracies))

optimal_k = k_values[np.argmax(mean_accuracies)]

print("Optimal number of neighbors:", optimal_k)
print("The average accuracy of cross-validation for the optimal number of neighbors:", np.max(mean_accuracies))

plt.plot(k_values, mean_accuracies, marker='o')
plt.title('Average accuracy of cross-validation in terms of number of neighbors')
plt.xlabel('Number of neighbors (k)')
plt.ylabel('Validation average accuracy')
plt.show()

knn_model_optimal_k_subset = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model_optimal_k_subset.fit(X_train[:, :2], y_train)

h = .02  
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_model_optimal_k_subset.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
plt.title('Decision boundary with kNN and optimal number of neighbors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
