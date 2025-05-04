# Iris Flower Classification using Decision Tree

##  Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Libraries Used](#libraries-used)
- [Model Details](#model-details)
- [Confusion Matrix & Accuracy](#confusion-matrix--accuracy)
- [Decision Tree Visualization](#decision-tree-visualization)
- [How to Run](#how-to-run)
- [Author](#author)

##  Project Overview
This project applies a **Decision Tree Classifier** to the famous Iris flower dataset. The goal is to classify flowers into three species (Setosa, Versicolor, Virginica) based on petal and sepal measurements.

##  Dataset Description
The dataset contains 150 samples with 4 features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

Target Classes:
- 0: Setosa
- 1: Versicolor
- 2: Virginica

##  Libraries Used
```python
pandas
numpy
matplotlib
sklearn
```

##  Model Details
- Algorithm: Decision Tree Classifier
- Criterion: Entropy
- Max Depth: 2
- Splitter: Best
- Train-Test Split: 70-30

##  Confusion Matrix & Accuracy
After training and predicting, the model generates a confusion matrix and prints the accuracy and classification report.

##  Decision Tree Visualization
A visual representation of the decision tree is displayed using `matplotlib.pyplot` and `sklearn.tree.plot_tree()`.

##  How to Run
1. Clone this repo or download the notebook.
2. Ensure dependencies are installed.
3. Run the Python script or notebook.

##  Author
Jaya Harshitha Mannela
