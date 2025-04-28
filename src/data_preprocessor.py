import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class DataPreprocessor:
    def __init__(self,df, language):
        """ 
        Initialize the data preprocessor
        """
        self.language = language
        self.df = df
    
    def get_dataset_stats(self):
        """ Return statistics about the dataset """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
            
        stats = {
            'total_samples': len(self.df),
            'unique_queries': self.df['query'].nunique(),
            'avg_products_per_query': len(self.df) / self.df['query'].nunique(),
            'relevance_distribution': self.df['relevance'].value_counts(normalize=True).sort_index().to_dict(),
        }
        
        if 'product_locale' in self.df.columns:
            stats['locale_distribution'] = self.df['product_locale'].value_counts(normalize=True).to_dict()
        
        return stats
    
    def get_bag_of_words(self, column='combined_text', min_df=2, max_features=100):
        """ Return a DataFrame with word frequencies

        Analyze text using Bag of Words and return a DataFrame with word frequencies.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        # Create combined text column if it doesn't exist
        if column == 'combined_text' and column not in self.df.columns:
            self.df['combined_text'] = self.df['title'].fillna('') + " " + self.df['description'].fillna('')
        
        # Get the text series
        text_series = self.df[column]
        
        # Remove empty texts
        valid_texts = text_series[text_series != ""].tolist()
        
        if not valid_texts:
            print(f"No valid texts found in {column}")
            return None
        
        # Create bag of words
        vectorizer = CountVectorizer(stop_words='english', min_df=min_df, max_features=max_features)
        X = vectorizer.fit_transform(valid_texts)
        
        # Get feature names and counts
        feature_names = vectorizer.get_feature_names_out()
        counts = X.sum(axis=0).A1
        
        # Create DataFrame with word frequencies
        word_freq = pd.DataFrame({'word': feature_names, 'frequency': counts})
        word_freq = word_freq.sort_values('frequency', ascending=False)
        
        return word_freq
    
    def split_data(self,split_column):
        """ Return train and test dataframes

        Splits the DataFrame into train, and test sets based on the split_column.
        """
        df_train = self.df[self.df[split_column] == 'train']
        df_test = self.df[self.df[split_column] == 'test']

        print(f"Train set size: {len(df_train)} rows, {df_train['query'].nunique()} unique queries")
        print(f"Test set size: {len(df_test)} rows, {df_test['query'].nunique()} unique queries")
        return df_train, df_test

    # Clean html tags for text analysis
    def remove_html_tags(self,text):
        """ Return a string with HTML tags removed"""
        if pd.isna(text) or text == 'None' or not isinstance(text, str):
            return ""
        # Remove HTML tags and special characters
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        return text.lower()
    # Clean and prepare text data
    def remove_special_characters(self,text):
        """ Return a string with special characters removed"""
        if pd.isna(text) or text == 'None' or not isinstance(text, str):
            return ""
        # Remove special characters and convert to lowercase
        return re.sub(r'[^\w\s]', ' ', text.lower())
    def data_preprocessing(self):
        """
        Preprocess the data by removing HTML tags and cleaning text.
        """
        # Remove HTML tags and clean text
        self.df['query'] = self.df['query'].apply(self.remove_html_tags)
        self.df['title'] = self.df['title'].apply(self.remove_html_tags)
        self.df['description'] = self.df['description'].apply(self.remove_html_tags)

        # Only remove special characters for US language
        if (self.language=='us'):
            self.df['query'] = self.df['query'].apply(self.remove_special_characters)
            self.df['title'] = self.df['title'].apply(self.remove_special_characters)
            self.df['description'] = self.df['description'].apply(self.remove_special_characters)

        self.df['combined_text'] = self.df['title'] + " " + self.df['description']

    def filter_by_locale(self):
        """
        Filter the dataset by product locale
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
            
        if 'product_locale' not in self.df.columns:
            raise ValueError("'product_locale' column not found in dataset")
            
        filtered_df = self.df[self.df['product_locale'] == self.language]
        
        print(f"Filtered dataset to {len(filtered_df)} rows with locale '{self.language}'")
        
        # Set the filtered DataFrame as the current DataFrame
        self.df = filtered_df