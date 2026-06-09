# Emmanuel-Moemeke-Deliverable---week-6---AnalystLab-Africa
Improving predictive performance using advanced machine learning algorithms and optimization techniques.
# Housing Price Prediction: Engineering for Stability 🏡

## Overview
This project builds a predictive machine learning model for housing prices using Python and Scikit-Learn. 

Instead of accepting a high-scoring baseline model that was secretly memorizing the data (overfitting), the primary focus of this project was to engineer a robust, production-ready system. By implementing a dynamic hyperparameter pipeline and rigorous cross-validation, the final algorithm guarantees high predictive accuracy on entirely unseen data.

## The Engineering Approach
The initial baseline Gradient Boosting Regressor scored highly on a single train/test split (Test R²: 96.39%), but lacked proven stability. 

To prevent the algorithm from memorizing noise and to ensure true generalization, I utilized `GridSearchCV` with 5-fold cross-validation to enforce the following constraints:
* Max Depth: `4` (Capped tree complexity)
* Learning Rate: `0.05` (Deliberate, slower pattern extraction)
* N_Estimators: `300` (Sequential trees to carefully correct residual errors)

## Final Performance Metrics
The tuned model sacrificed a microscopic fraction of raw test accuracy to achieve guaranteed stability across multiple data shuffles:

* Training R² Score: 99.95%
* Cross-Validation R² Score: 96.08%
* Final Test R² Score: 96.20%

*Conclusion: The model reliably explains ~96.2% of the variance in housing prices on unseen data, proving its effectiveness as a stable, real-world predictive tool.*

## Tech Stack
* Language: Python
* Machine Learning: Scikit-Learn (`GradientBoostingRegressor`, `Pipeline`, `GridSearchCV`)
* Data Manipulation: Pandas, NumPy
* Visualization: Matplotlib, Seaborn

   ```bash
   git clone [https://github.com/yourusername/housing-price-prediction.git](https://github.com/yourusername/housing-price-prediction.git)
