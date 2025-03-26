# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_explore_data(file_path):
    """
    Load the dataset and perform initial exploration
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    # Load the dataset
    reviews = pd.read_csv(file_path)
    
    # Print column names
    print("Column Names:")
    print(reviews.columns.tolist())
    
    # Print dataset information
    print("\nDataset Information:")
    reviews.info()
    
    return reviews

def transform_binary_feature(df, column):
    """
    Transform a binary feature to 0 and 1
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Name of the column to transform
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with transformed column
    """
    # Check original value counts
    print(f"\nOriginal {column} Value Counts:")
    print(df[column].value_counts())
    
    # Create binary dictionary
    binary_dict = {False: 0, True: 1}
    
    # Transform column
    df[column] = df[column].map(binary_dict)
    
    # Verify transformation
    print(f"\nTransformed {column} Value Counts:")
    print(df[column].value_counts())
    
    return df

def transform_ordinal_feature(df, column, mapping):
    """
    Transform an ordinal feature using a provided mapping
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Name of the column to transform
    mapping : dict
        Dictionary for mapping values
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with transformed column
    """
    # Check original value counts
    print(f"\nOriginal {column} Value Counts:")
    print(df[column].value_counts())
    
    # Transform column
    df[column] = df[column].map(mapping)
    
    # Verify transformation
    print(f"\nTransformed {column} Value Counts:")
    print(df[column].value_counts())
    
    return df

def one_hot_encode_categorical(df, column):
    """
    Perform one-hot encoding on a categorical feature
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Name of the column to one-hot encode
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with one-hot encoded columns
    """
    # Check original categories
    print(f"\nOriginal {column} Categories:")
    print(df[column].value_counts())
    
    # Perform one-hot encoding
    one_hot = pd.get_dummies(df[column], prefix=column)
    
    # Join the new columns back to the original dataframe
    df = pd.concat([df, one_hot], axis=1)
    
    # Print updated column names
    print("\nUpdated Column Names:")
    print(df.columns.tolist())
    
    return df

def transform_datetime_feature(df, column):
    """
    Transform a datetime feature to proper datetime type
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Name of the datetime column
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with transformed datetime column
    """
    # Transform to datetime
    df[column] = pd.to_datetime(df[column])
    
    # Extract additional datetime features
    df[f'{column}_year'] = df[column].dt.year
    df[f'{column}_month'] = df[column].dt.month
    df[f'{column}_day'] = df[column].dt.day
    df[f'{column}_dayofweek'] = df[column].dt.dayofweek
    
    # Verify transformation
    print(f"\n{column} Data Type:")
    print(df[column].dtype)
    
    return df

def scale_numerical_features(df, columns_to_scale):
    """
    Scale numerical features using StandardScaler
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns_to_scale : list
        List of column names to scale
    
    Returns:
    --------
    numpy.ndarray
        Scaled feature matrix
    """
    # Select numerical features
    numerical_df = df[columns_to_scale].copy()
    
    # Instantiate StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(numerical_df)
    
    # Print first 5 rows of scaled data
    print("\nFirst 5 rows of scaled data:")
    print(scaled_data[:5])
    
    # Print mean and standard deviation of scaled data
    print("\nMean of scaled features:")
    print(np.mean(scaled_data, axis=0))
    print("\nStandard Deviation of scaled features:")
    print(np.std(scaled_data, axis=0))
    
    return scaled_data

def main():
    """
    Main function to orchestrate data transformation process
    """
    # Load the dataset
    reviews = load_and_explore_data('reviews.csv')
    
    # Transform binary feature (recommended)
    reviews = transform_binary_feature(reviews, 'recommended')
    
    # Transform date feature (review_date)
    reviews = transform_datetime_feature(reviews, 'review_date')
    
    # One-hot encode department name
    reviews = one_hot_encode_categorical(reviews, 'department_name')
    
    # One-hot encode division name (if needed)
    reviews = one_hot_encode_categorical(reviews, 'division_name')
    
    # Select numerical features for scaling
    numerical_columns = [
        'clothing_id', 'age', 
        'recommended', 
        'review_date_year', 'review_date_month', 'review_date_day', 'review_date_dayofweek'
    ]
    
    # Get one-hot encoded department columns
    department_columns = [col for col in reviews.columns if col.startswith('department_name_')]
    division_columns = [col for col in reviews.columns if col.startswith('division_name_')]
    
    # Combine numerical columns with categorical columns
    columns_to_scale = numerical_columns + department_columns + division_columns
    
    # Scale numerical features
    scaled_data = scale_numerical_features(reviews, columns_to_scale)
    
    return reviews, scaled_data

# Run the main function
if __name__ == '__main__':
    transformed_reviews, scaled_reviews = main()