import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def preprocess_employee_data(df):
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Handle missing values - avoid chained assignment by direct assignment
    df_clean = df_clean.assign(
        salary=lambda x: x['salary'].fillna(x['salary'].mean()),
        department=lambda x: x['department'].fillna('Unknown')
    )
    
    # Convert joining_date to datetime first, then handle missing values
    df_clean['joining_date'] = pd.to_datetime(df_clean['joining_date'], format='mixed', errors='coerce')
    # If no valid dates exist, use today's date as default, otherwise use most common date
    default_date = df_clean['joining_date'].mode()[0] if not df_clean['joining_date'].isna().all() else pd.Timestamp.today()
    df_clean['joining_date'] = df_clean['joining_date'].fillna(default_date)
    
    # Standardize department names
    department_mapping = {
        'hr': 'HR',
        'human resources': 'HR',
        'it': 'IT',
        'information technology': 'IT',
        'sales': 'Sales',
        'marketing': 'Marketing',
        'unknown': 'Unknown'
    }
    
    # Normalize to lowercase, map to standardized names, and ensure unmapped values become 'Unknown'
    df_clean['department'] = df_clean['department'].str.lower()
    df_clean['department'] = df_clean['department'].map(department_mapping).fillna('Unknown')
    
    # Encode categorical variables
    le_dept = LabelEncoder()
    le_role = LabelEncoder()

    # Handle missing job roles through direct assignment
    df_clean = df_clean.assign(job_role=lambda x: x['job_role'].fillna('Unknown'))
    
    # Encode categorical variables
    df_clean = df_clean.assign(
        department_encoded=lambda x: le_dept.fit_transform(x['department']),
        job_role_encoded=lambda x: le_role.fit_transform(x['job_role'])
    )
    
    return df_clean

# Example usage
if __name__ == "__main__":
    # Sample data (you would typically read this from a file)
    data = {
        'salary': [50000, None, 75000],
        'department': ['HR', 'human resources', None],
        'joining_date': ['2022-01-01', None, '2022/03/15'],
        'job_role': ['Manager', 'Developer', 'Analyst']
    }
    
    df = pd.DataFrame(data)
    cleaned_df = preprocess_employee_data(df)
    print(cleaned_df)