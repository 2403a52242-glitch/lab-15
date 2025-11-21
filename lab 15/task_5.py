import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_financial_data(df):
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # 1. Handle missing values
    numeric_imputer = SimpleImputer(strategy='mean')
    numeric_columns = ['price', 'volume']
    df_processed[numeric_columns] = numeric_imputer.fit_transform(df_processed[numeric_columns])
    
    # 2. Create technical indicators
    # 7-day moving average
    df_processed['MA7'] = df_processed['price'].rolling(window=7).mean()
    # 30-day moving average
    df_processed['MA30'] = df_processed['price'].rolling(window=30).mean()
    # Trading volume moving average
    df_processed['volume_MA7'] = df_processed['volume'].rolling(window=7).mean()
    
    # Calculate price volatility
    df_processed['volatility'] = df_processed['price'].rolling(window=7).std()
    
    # 3. Normalize continuous variables
    scaler = StandardScaler()
    columns_to_normalize = ['price', 'volume', 'MA7', 'MA30', 'volume_MA7', 'volatility']
    df_processed[columns_to_normalize] = scaler.fit_transform(
        df_processed[columns_to_normalize].fillna(0)
    )
    
    # 4. Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['sector', 'company_name']
    for col in categorical_columns:
        if col in df_processed.columns:
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
    
    # Fill any remaining NaN values with 0
    df_processed = df_processed.fillna(0)
    
    return df_processed

# Example usage
if __name__ == "__main__":
    # Sample data creation
    dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'price': np.random.uniform(10, 100, len(dates)),
        'volume': np.random.uniform(1000, 10000, len(dates)),
        'sector': np.random.choice(['Tech', 'Finance', 'Healthcare'], len(dates)),
        'company_name': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], len(dates))
    })
    
    # Process the data
    processed_df = preprocess_financial_data(sample_data)
    print("Processed DataFrame Shape:", processed_df.shape)
    print("\nFirst few rows of processed data:")
    print(processed_df.head())