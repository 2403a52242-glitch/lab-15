import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# Some NLTK tokenizers require the 'punkt_tab' resource on certain installs
# download it proactively to avoid LookupError at runtime
nltk.download('punkt_tab')

def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

def process_dataset(input_file, output_file):
    try:
        # Read the dataset (assuming CSV format with 'text' column)
        df = pd.read_csv(input_file)
        
        # Apply preprocessing to the text column
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Save the processed dataset
        df.to_csv(output_file, index=False)
        print(f"Processed dataset saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_file = "social_media_data.csv"
    output_file = "processed_social_media_data.csv"
    
    process_dataset(input_file, output_file)