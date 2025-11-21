import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'Paris', 'London', 'Tokyo']
}

# Create DataFrame
df = pd.DataFrame(data)

# Basic operations
print("Original DataFrame:")
print(df)

# Display basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Sort by Age
print("\nSorted by Age:")
print(df.sort_values('Age'))

# Filter data
print("\nPeople older than 30:")
print(df[df['Age'] > 30])