
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import learning_curve

def plot_decision_boundary(model, X, y, label_encoder=None, title="Decision Boundary", figsize=(7,5)):
    X = X.values if hasattr(X, "values") else X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=figsize)
    cmap = ListedColormap(sns.color_palette("viridis", len(np.unique(y))))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=cmap)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    if label_encoder:
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              label=label_encoder.inverse_transform([i])[0],
                              markerfacecolor=cmap(i), markersize=8) for i in range(len(label_encoder.classes_))]
        plt.legend(handles=handles, title="Classes")
    plt.tight_layout()
    plt.show()

def plot_learning_curve(estimator, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), figsize=(7, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_scores_mean, label="Training score", marker="o")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", marker="o")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel(scoring.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
