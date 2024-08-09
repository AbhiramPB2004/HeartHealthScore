# Heart Health Prediction

## Overview

Welcome to the Heart Health Prediction project! This project leverages machine learning techniques to predict heart health based on various physiological and clinical parameters. Using Scikit-learn for modeling, Pandas for data handling, and Plotly for visualization, this project aims to provide an easy-to-use tool for assessing heart health.

The project is implemented as a Jupyter Notebook, which provides an interactive environment for running the code and visualizing the results.

## Project Structure

The project is organized into the following sections in the Jupyter Notebook:

- **Data Preparation**: Code cells for cleaning and preparing the dataset.
- **Model Training**: Code cells for training machine learning models.
- **Evaluation**: Code cells to evaluate model performance and accuracy.
- **Visualization**: Code cells to visualize results and insights using Plotly.
- **Main Application**: The notebook itself where you interact with the model and make predictions.

## Parameters

The model uses the following parameters to predict heart health:

1. **Age**: Age of the individual.
2. **Sex**: Gender of the individual.
3. **CP**: Chest pain type.
4. **Trestbps**: Resting blood pressure.
5. **Chol**: Serum cholesterol.
6. **Fbs**: Fasting blood sugar.
7. **Restecg**: Resting electrocardiographic results.
8. **Thalach**: Maximum heart rate achieved.
9. **Exang**: Exercise induced angina.
10. **Oldpeak**: Depression induced by exercise relative to rest.
11. **Slope**: Slope of the peak exercise ST segment.
12. **Ca**: Number of major vessels colored by fluoroscopy.
13. **Thal**: Thalassemia.
14. **Target**: Heart disease presence (0 = No, 1 = Yes).
15. **High_col**: A custom feature to denote high cholesterol levels (derived from the Chol parameter).

## RandomForestRegressor Parameters

In this project, we use `RandomForestRegressor` with the following parameters:

- `bootstrap`: `True` - Whether bootstrap samples are used when building trees.
- `ccp_alpha`: `0.0` - Complexity parameter used for Minimal Cost-Complexity Pruning.
- `class_weight`: `None` - Weights associated with classes.
- `criterion`: `'gini'` - The function to measure the quality of a split (used for classification, but here for regression it's `'squared_error'`).
- `max_depth`: `None` - The maximum depth of the tree.
- `max_features`: `'sqrt'` - The number of features to consider when looking for the best split.
- `max_leaf_nodes`: `None` - Grow trees with max_leaf_nodes in best-first fashion.
- `max_samples`: `None` - If bootstrap is True, the number of samples to draw from X to train each base estimator.
- `min_impurity_decrease`: `0.0` - A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
- `min_samples_leaf`: `1` - The minimum number of samples required to be at a leaf node.
- `min_samples_split`: `2` - The minimum number of samples required to split an internal node.
- `min_weight_fraction_leaf`: `0.0` - The minimum weighted fraction of the sum total of weights required to be at a leaf node.
- `n_estimators`: `100` - The number of trees in the forest.
- `n_jobs`: `None` - The number of jobs to run in parallel for both `fit` and `predict`.
- `oob_score`: `False` - Whether to use out-of-bag samples to estimate the generalization score.
- `random_state`: `None` - Controls the randomness of the estimator.
- `verbose`: `0` - Controls the verbosity when fitting and predicting.
- `warm_start`: `False` - Whether to reuse the solution of the previous call to `fit` and add more estimators to the ensemble.

## Installation

To get started with the Heart Health Prediction project, follow the instructions below:

1. Clone the repository:
    ```bash
    git clone https://github.com/AbhiramPB2004/HeartHealthScore.git
    ```

2. Navigate into the project directory:
    ```bash
    cd HeartHealthScore
    ```

3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

6. Open the notebook file (`HeartHealthScore.ipynb`) from the Jupyter interface.

## Usage

In the Jupyter Notebook, you can run each cell to prepare the data, train the model, and make predictions. Follow the instructions within the notebook to input parameters and get heart health predictions.

## Dependencies

This project uses the following Python libraries:
- `scikit-learn` for machine learning algorithms
- `pandas` for data manipulation
- `plotly` for interactive visualizations

You can find the specific versions of these libraries in the `requirements.txt` file.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Add your message here"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Open a pull request on GitHub.

## Acknowledgments

- Thanks to the Scikit-learn, Pandas, and Plotly communities for their invaluable tools and libraries.
- Special thanks to the dataset contributors and researchers in the field of heart disease prediction.


