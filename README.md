# Iris Dataset Classification using kNN and SVM

## 📌 Overview
This project compares two classic machine learning algorithms — **k-Nearest Neighbors (kNN)** and **Support Vector Machine (SVM)** — for classification tasks on the Iris dataset. It demonstrates model training, cross-validation, hyperparameter tuning, and visualization of decision boundaries.

Key aspects include:
- Training and evaluation of kNN models (with and without weighting)
- Cross-validation to determine the optimal value of `k`
- Visualizing decision boundaries using top two features
- SVM classification using RBF kernel and comparison with kNN

## 📊 Technologies Used
- Python
- Scikit-learn
- NumPy
- Matplotlib

## 📁 Project Structure
```
src/
    Q1.py                  → kNN classifier implementation with cross-validation and visualization
    Q2.py                  → SVM classifier implementation with decision boundary plot
```

## ▶️ How to Run

1. Clone the repository:
```
git clone https://github.com/mohammadbaghershahmir/iris-classification-knn-svm.git
```

2. Install dependencies:
```
pip install scikit-learn numpy matplotlib
```

3. Run the scripts:
```bash
cd src
python Q1.py
python Q2.py
```

## 📈 Sample Output
- Accuracy of kNN and weighted kNN
- Optimal `k` via cross-validation
- Visualization of decision boundaries for both kNN and SVM

## 🏷️ Tags
`machine-learning` `iris-dataset` `knn` `svm` `classification` `scikit-learn`

## 📄 License
MIT
