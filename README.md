# Iris-Flower-Classification-task


## 🚀 Features

- 📊 **Exploratory Data Analysis (EDA)**: Understanding data distributions, handling duplicates, checking for null values, and visualizing correlations.
- 📉 **Data Visualization**: Histograms, pairplots, boxplots, heatmaps using Seaborn and Matplotlib.
- 🛠️ **Data Preprocessing**: Handling duplicates, encoding categorical data, feature scaling.
- 🤖 **Model Building**:
  - Logistic Regression 
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Naive Bayes Classifier
  - Neural network (Multi-layer Perceptron)
- 🏆 **Model Evaluation**:
  - Confusion Matrix
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-Score)
- 🔥 **Hyperparameter Tuning**:
  - GridSearchCV / Manual tuning for various models
  - Learning rate and optimizer tuning in PyTorch models
- ⏳ **Advanced Techniques**:
  - Early Stopping & Learning Rate Scheduling (in PyTorch MLP)
  - Dropout layers to prevent overfitting

## 📁 Dataset

- **Source**: CSV file as attached in the repository
- **Columns**:
  - SepalLengthCm
  - SepalWidthCm
  - PetalLengthCm
  - PetalWidthCm
  - Species (Target variable)

## 📌 Dependencies

Install the following libraries:
pip install numpy pandas matplotlib seaborn scikit-learn torch

## 📝 Usage

1. Clone the repository:
   git clone https://github.com/namanjain2020/Iris-Flower-Classification-task.git
   cd iris-flower-classification

2. Run the notebook:
   jupyter notebook iris_flower.ipynb
  

3. Follow the notebook for step-by-step execution:
   - Data exploration
   - Visualization
   - Model training
   - Evaluation & tuning

## 📈 Results

- Achieved high classification accuracy across models, particularly with **Neural Network(MLP)** and the best CV score by **Support Vector Machine(SVM)**.
- Demonstrated the impact of hyperparameter tuning, early stopping, and learning rate scheduling on model performance.
