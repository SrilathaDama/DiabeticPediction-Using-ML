### Diabetes Prediction Using Machine Learning

#### Overview
This project aims to develop a machine learning model to predict diabetes based on various health indicators. The model utilizes an ensemble approach to enhance prediction accuracy and is evaluated using metrics like accuracy, precision, recall, F1 Score, and AUC-ROC.

#### Features
- **Data Preprocessing:** Cleaning and preparing the dataset for model training.
- **Model Development:** Implementing and fine-tuning various machine learning algorithms.
- **Ensemble Learning:** Combining multiple models to improve prediction performance.
- **Evaluation Metrics:** Assessing the model's performance using accuracy, precision, recall, F1 Score, and AUC-ROC.

#### Technologies Used
- **Programming Languages:** Python
- **Libraries:** Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn
- **Tools:** Jupyter Notebook

#### Project Structure
```
Diabetes_Prediction/
│
├── data/
│   ├── diabetes.csv               # Dataset file
│
├── models/
│   ├── knn_model.pkl              # Trained K-Nearest Neighbors model
│   ├── adaboost_model.pkl         # Trained AdaBoost model
│   ├── decision_tree_model.pkl    # Trained Decision Tree model
│
├── notebooks/
│   ├── Diabetes_Prediction.ipynb  # Jupyter Notebook containing the project code
│
├── results/
│   ├── evaluation_metrics.txt     # File containing model evaluation metrics
│   ├── roc_curve.png              # ROC curve plot
│
├── README.md                      # Project README file
└── requirements.txt               # List of dependencies
```

#### How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/SrilathaDama/ML-Project.git
   cd ML-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   Open `Diabetes_Prediction.ipynb` in Jupyter Notebook and run all cells to see the results.

#### Results
- **Accuracy:** 95%
- **AUC-ROC Score:** 98%
- **Precision, Recall, F1 Score:** Detailed in the `evaluation_metrics.txt` file in the `results` directory.

#### Conclusion
This project demonstrates the effectiveness of using an ensemble approach for diabetes prediction, achieving high accuracy and robustness. The model can be further improved and extended to include more features and advanced techniques.

#### Future Work
- **Feature Engineering:** Explore additional features to improve model performance.
- **Advanced Models:** Implement deep learning models for potentially better results.
- **Deployment:** Develop a web or mobile application to make the model accessible to a broader audience.
