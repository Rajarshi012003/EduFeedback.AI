import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)

def load_student_data(file_path):
    """
    Load the Portuguese student dataset and perform initial preprocessing.
    
    Args:
        file_path (str): Path to the student-merged.csv file
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def preprocess_student_data(df):
    """
    Preprocess the Portuguese student dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Make a copy to avoid modifying the original data
    df = df.copy()
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Impute missing values
    if len(numeric_columns) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    
    if len(categorical_columns) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Calculate derived features
    df = calculate_derived_features(df)
    
    return df, label_encoders

def calculate_derived_features(df):
    """
    Calculate derived features from the student dataset.
    
    Args:
        df (pd.DataFrame): Dataset with basic preprocessing applied
        
    Returns:
        pd.DataFrame: Dataset with additional derived features
    """
    df = df.copy()
    
    # Academic performance features
    # Average of first period grades
    grade_cols_x = ['G1_x', 'G2_x', 'G3_x']
    grade_cols_y = ['G1_y', 'G2_y', 'G3_y']
    
    # Calculate average performance across subjects
    df['avg_grade_x'] = df[grade_cols_x].mean(axis=1)
    df['avg_grade_y'] = df[grade_cols_y].mean(axis=1)
    df['overall_avg_grade'] = df[grade_cols_x + grade_cols_y].mean(axis=1)
    
    # Performance improvement indicators
    df['improvement_x'] = df['G3_x'] - df['G1_x']
    df['improvement_y'] = df['G3_y'] - df['G1_y']
    
    # Study and social factors
    # Combine study time from both subjects
    df['total_studytime'] = df['studytime_x'] + df['studytime_y']
    df['total_failures'] = df['failures_x'] + df['failures_y']
    df['total_absences'] = df['absences_x'] + df['absences_y']
    
    # Family support indicators
    df['family_support_score'] = (
        df['famsup_x'] + df['famsup_y'] + 
        df['famrel_x'] + df['famrel_y']
    ) / 4
    
    # Social activity score
    df['social_activity_score'] = (
        df['freetime_x'] + df['freetime_y'] + 
        df['goout_x'] + df['goout_y']
    ) / 4
    
    # Alcohol consumption score
    df['alcohol_consumption'] = (
        df['Dalc_x'] + df['Dalc_y'] + 
        df['Walc_x'] + df['Walc_y']
    ) / 4
    
    # Health and wellness score
    df['health_score'] = (df['health_x'] + df['health_y']) / 2
    
    # Educational support score
    df['educational_support'] = (
        df['schoolsup_x'] + df['schoolsup_y'] + 
        df['paid_x'] + df['paid_y']
    ) / 4
    
    # Engagement score (composite of multiple factors)
    # Normalize components first
    scaler = StandardScaler()
    engagement_components = [
        'total_studytime', 'family_support_score', 
        'educational_support', 'health_score'
    ]
    
    # Only include components that exist and have non-null values
    valid_components = [col for col in engagement_components if col in df.columns and not df[col].isnull().all()]
    
    if valid_components:
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[valid_components]),
            columns=valid_components,
            index=df.index
        )
        df['engagement_score'] = df_scaled.mean(axis=1)
    else:
        df['engagement_score'] = 0
    
    # Performance consistency
    df['performance_consistency_x'] = df[grade_cols_x].std(axis=1)
    df['performance_consistency_y'] = df[grade_cols_y].std(axis=1)
    
    # Risk factors
    df['risk_score'] = (
        df['total_failures'] * 0.3 + 
        df['alcohol_consumption'] * 0.2 + 
        df['total_absences'] * 0.2 + 
        (5 - df['health_score']) * 0.3  # Invert health score
    )
    
    return df

def normalize_features(df):
    """
    Normalize numerical features in the dataset.
    
    Args:
        df (pd.DataFrame): Dataset with derived features
        
    Returns:
        pd.DataFrame: Dataset with normalized features
    """
    df = df.copy()
    
    # Select numerical columns for normalization
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude target variables and IDs from normalization
    exclude_columns = [
        'G1_x', 'G2_x', 'G3_x', 'G1_y', 'G2_y', 'G3_y',
        'avg_grade_x', 'avg_grade_y', 'overall_avg_grade'
    ]
    
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    if numeric_columns:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

def remove_duplicates(df):
    """
    Remove duplicate entries from the dataset.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset without duplicates
    """
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    
    if initial_shape[0] > final_shape[0]:
        print(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows.")
    
    return df

def main():
    # Path to the dataset
    data_path = 'student-merged.csv'
    
    # Load and preprocess the data
    df = load_student_data(data_path)
    
    # Preprocess the data
    df, label_encoders = preprocess_student_data(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Normalize features
    df = normalize_features(df)
    
    # Save the processed data
    df.to_csv('data/processed/student_processed.csv', index=False)
    
    print("Data preprocessing completed.")
    print(f"Processed data saved to 'data/processed/student_processed.csv'")
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
