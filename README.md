# Bike Sharing Demand Prediction 🚴‍♂️

![Bike Sharing](https://cdn.dribbble.com/users/213322/screenshots/1375431/bike_animated.gif)

This project focuses on predicting the demand for bike-sharing services using **supervised machine learning** and **deep learning** techniques. The dataset involves **multi-label regression**, predicting multiple target variables that include hourly and daily counts of rental bikes.

---

## Overview

The **Bike Sharing Demand Prediction** project aims to analyze and predict bike demand based on temporal, environmental, and seasonal factors. It uses advanced regression techniques and compares model performance to identify the most accurate and efficient approach.

---

### Key Objectives
- Build regression models for **multi-label prediction** (hourly and daily demand).
- Evaluate and compare machine learning and deep learning methods.
- Fine-tune models to achieve optimal prediction accuracy.

---

## Dataset

The dataset is sourced from Kaggle: [Bike Sharing Dataset](https://www.kaggle.com/c/bike-sharing-demand). It includes features such as:
- **Datetime**: Timestamp of bike usage.
- **Season**: Weather seasons (spring, summer, etc.).
- **Weather Condition**: Weather codes indicating the condition (e.g., clear, cloudy, etc.).
- **Temperature**: Scaled temperature in Celsius.
- **Humidity**: Normalized relative humidity.
- **Windspeed**: Normalized wind speed.
- **Count**: Total number of bikes rented.

### Dataset Preparation

#### Data Cleaning and Preprocessing:
1. **Datetime Feature Engineering**:
   - Extracted features like **hour**, **day**, **month**, and **year**.
   - Used cyclical encoding for **hour** and **day** to capture patterns.
2. **Handling Missing Values**:
   - Imputed missing weather and wind speed values.
3. **Scaling**:
   - Applied **Robust Scaler** to reduce the influence of outliers on numerical features.
4. **Encoding Categorical Features**:
   - Used **one-hot encoding** for season and weather condition.

---

## Methodology

### Model Exploration

1. **Linear Regression**  
   - Baseline regression model for predicting bike demand.  
   - Provided insights into linear trends between features and labels.

2. **Ridge Regression**  
   - Added L2 regularization to penalize high weights, reducing overfitting.

3. **Lasso Regression**  
   - Introduced L1 regularization to enforce sparsity and improve interpretability.

4. **K-Nearest Neighbors (KNN)**  
   - Predicted demand by considering the nearest data points in feature space.

5. **Decision Tree**  
   - Captured feature interactions using tree-based structures.  
   - High interpretability but prone to overfitting.

6. **Random Forest**  
   - An ensemble model of decision trees for better generalization and reduced overfitting.

---

### Deep Learning  
Built a **Neural Network** for demand prediction with the following architecture:
- **Input Layer**: Processed temporal and environmental features.
- **Hidden Layers**:
  - Dense layers with **ReLU activation**.
  - Dropout layers for regularization.
- **Output Layer**: Multi-label output predicting both **hourly** and **daily counts** using a **linear activation** function.


---

## Technologies Used
- **Python**: Core programming language.
- **pandas**: Data manipulation and preprocessing.
- **numpy**: Numerical computations.
- **scikit-learn**: Machine learning models.
- **tensorflow/keras**: Deep learning framework.
- **matplotlib/plotly**: Data visualization.

---

## Getting Started

### Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/youssefa7med/BikeSharing.git
   cd BikeSharing
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset in the `data` folder.

---

### Run the Project
1. **Train Models**:  
   ```bash
   python train.py
   ```

2. **Evaluate Models**:  
   ```bash
   python evaluate.py
   ```

3. **Visualize Results**:  
   Use `visualize_results.ipynb` for interactive result analysis.

---

## Contributing
Contributions are welcome! Feel free to fork the repo and submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
Special thanks to Kaggle for providing the dataset and the open-source community for the tools used in this project.
