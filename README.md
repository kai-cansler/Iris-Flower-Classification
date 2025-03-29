# 🌸 Iris Flower Classification

## 📌 Overview

This project uses various machine learning algorithms to classify iris flowers into three species:

- **Setosa**
- **Versicolor**
- **Virginica**

We explore and compare five different classifiers on the classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html), focusing on **accuracy**, **decision boundaries**, and **interpretability**. The dataset contains 150 samples (50 per class) and four features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## 🤖 Models Compared

Five classification models were trained and evaluated:

- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVC, RBF kernel)
- Logistic Regression
- Gaussian Naive Bayes

We visualized each model's decision boundary using only two features (*petal length* and *petal width*) for clarity. Each classifier was evaluated with **train/test accuracy** and **confusion matrices**. Misclassified points were highlighted to provide insight into model behavior.

📌 **Key Takeaway:**  
KNN and SVC performed best in terms of accuracy, but each model presents tradeoffs in flexibility, speed, and interpretability.

## 📊 Dataset & Target Mapping

- **Features (Inputs):**  
  `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm`

- **Target (Output):**  
  - `0` → Setosa  
  - `1` → Versicolor  
  - `2` → Virginica

## 🚀 Methodology

1. **Data Preprocessing**
   - Load and explore the Iris dataset
   - Normalize features (if needed)
   - Encode class labels
   - Train/test split

2. **Model Training & Tuning**
   - Train KNN and other classifiers
   - Tune hyperparameters (e.g., `k` for KNN)

3. **Evaluation**
   - Accuracy scores
   - Confusion matrices
   - Classification reports
   - Decision boundary visualization

## 📈 Results

- Best model accuracy: **~97%**
- Best value of K: **5**
- Full visual comparison of all models using a consistent viridis color palette
- Misclassified points shown for deeper analysis

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn

## 📂 Project Structure

📁 Iris-Classification-KNN  
│── 📂 data/                # Dataset (if manually provided)  
│── 📂 notebooks/           # Jupyter Notebooks for analysis  
│── 📂 models/              # Saved trained models (if applicable)  
│── iris_knn.py            # Main script for classification  
│── requirements.txt       # Dependencies  
│── README.md              # Project documentation  

## 🔧 Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/Iris-Classification-KNN.git
cd Iris-Classification-KNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the classification script
python iris_knn.py

##📈 Results

	•	Model accuracy: (e.g., 95%)
	•	Best value of K: (e.g., 3 or 5 based on tuning results)
	•	Confusion Matrix and Classification Report for performance analysis

##🌱 Future Improvements

	•	Add cross-validation and grid search
	•	Test more advanced models (e.g., ensemble methods, neural nets)
	•	Deploy using Flask or FastAPI
	•	Expand to multiclass visualizations in higher dimensions

##👨‍💻 Author

	•	Your Name
	•	GitHub: kai-cansler
