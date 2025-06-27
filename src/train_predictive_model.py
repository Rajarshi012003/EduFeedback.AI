import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Create necessary directories
os.makedirs('models/predictive', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

def load_processed_data(file_path):
    """
    Load the processed dataset.
    
    Args:
        file_path (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print(f"Loading processed data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def prepare_features_and_target(df):
    """
    Prepare features and target variable for modeling.
    
    Args:
        df (pd.DataFrame): Processed dataset
        
    Returns:
        tuple: Features (X), target (y), and feature column names
    """
    # Use G3_y as the target (final grade in second subject)
    target_column = 'G3_y'
    
    if target_column not in df.columns:
        # Fallback targets
        possible_targets = ['G3_x', 'overall_avg_grade', 'avg_grade_y']
        for target in possible_targets:
            if target in df.columns:
                target_column = target
                break
        else:
            raise ValueError("No suitable target column found")
    
    print(f"Using '{target_column}' as the target variable.")
    
    # Exclude target and related grade columns from features
    exclude_columns = [
        target_column, 'G1_y', 'G2_y', 'G3_y',  # Target subject grades
        'G1_x', 'G2_x', 'G3_x',  # Other subject grades (to avoid data leakage)
        'avg_grade_x', 'avg_grade_y', 'overall_avg_grade'  # Derived grade features
    ]
    
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target statistics - Mean: {y.mean():.2f}, Std: {y.std():.2f}, Range: [{y.min()}, {y.max()}]")
    
    return X, y, feature_columns

def train_and_evaluate_models(X, y):
    """
    Train and evaluate multiple models to find the best performer.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        
    Returns:
        tuple: Best model, model scores dictionary, and feature importances
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    
    # Define models to evaluate
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    # Dictionary to store model scores
    model_scores = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create a pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store scores
        model_scores[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'pipeline': pipeline
        }
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    # Find the best model based on R2 score
    best_model_name = max(model_scores, key=lambda k: model_scores[k]['R2'])
    best_model = model_scores[best_model_name]['pipeline']
    
    print(f"\nBest model: {best_model_name} with R2 score: {model_scores[best_model_name]['R2']:.4f}")
    
    # Get feature importances for Random Forest
    feature_importances = None
    if best_model_name == 'Random Forest':
        model_step = best_model.named_steps['model']
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model_step.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    return best_model, model_scores, feature_importances

def tune_random_forest(X, y):
    """
    Perform hyperparameter tuning for the Random Forest model.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        
    Returns:
        Pipeline: Tuned model pipeline
    """
    print("\nPerforming hyperparameter tuning for Random Forest...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    
    # Define the parameter grid
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, 20, None],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2]
    }
    
    # Create the grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation R2 score: {best_score:.4f}")
    
    # Evaluate on the test set
    y_pred = grid_search.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test set R2 score: {test_r2:.4f}")
    print(f"Test set RMSE: {test_rmse:.4f}")
    
    return grid_search.best_estimator_

def visualize_feature_importance(feature_importances, output_path):
    """
    Visualize feature importances and save the plot.
    
    Args:
        feature_importances (pd.DataFrame): DataFrame with feature importances
        output_path (str): Path to save the visualization
    """
    if feature_importances is None:
        print("No feature importances available for visualization.")
        return
    
    # Plot the top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
    plt.title('Top 15 Feature Importances - Student Performance Prediction')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Feature importance visualization saved to {output_path}")

def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained model
        file_path (str): Path to save the model
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def main():
    # Path to the processed dataset
    data_path = 'data/processed/student_processed.csv'
    
    # Check if the processed data exists
    if not os.path.exists(data_path):
        print("Processed data not found. Running data preprocessing...")
        import data_preprocessing
        data_preprocessing.main()
    
    # Load the processed data
    df = load_processed_data(data_path)
    
    # Prepare features and target
    X, y, feature_columns = prepare_features_and_target(df)
    
    # Train and evaluate models
    best_model, model_scores, feature_importances = train_and_evaluate_models(X, y)
    
    # Tune the Random Forest model
    tuned_model = tune_random_forest(X, y)
    
    # Use the tuned model as the best model
    best_model = tuned_model
    
    # Visualize feature importances if available
    if feature_importances is not None:
        visualize_feature_importance(feature_importances, 'visualizations/feature_importance.png')
    
    # Save the best model
    save_model(best_model, 'models/predictive/best_model.pkl')
    
    print("\nPredictive model training completed.")
    print("Model can predict student final grades (G3_y) based on demographic, social, and academic factors.")

if __name__ == "__main__":
    main()
