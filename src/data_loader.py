import pandas as pd

class DataLoader:
    def __init__(self, example_path,products_path,small_version=True,dataset_size_ratio=1):
        """
        Initialize the DataLoader with paths to the datasets."""
        self.example_path = example_path
        self.products_path = products_path
        self.small_version = small_version
        self.df = None
        self.dataset_size_ratio = dataset_size_ratio

    def load_data(self):
        """
        Load the datasets from the specified paths.
        """
        df_examples = pd.read_parquet(self.example_path)
        df_products = pd.read_parquet(self.products_path)

        # Merge the examples and products dataframes on product_locale and product_id
        self.df = pd.merge(
            df_examples,
            df_products,
            how='left',
            left_on=['product_locale','product_id'],
            right_on=['product_locale', 'product_id']
        )

        # Check if the user want to use a smaller dataset 
        if (self.dataset_size_ratio < 1 ):
            # Get unique queries 
            unique_queries = self.df['query'].unique()
            
            # Calculate how many unique queries to sample
            n_queries_to_sample = int(len(unique_queries) * self.dataset_size_ratio)
            
            # Randomly sample the queries
            sampled_queries = pd.Series(unique_queries).sample(
                n=n_queries_to_sample, 
                random_state=42  # For reproducibility
            ).values
            
            # Filter the dataframe to only include rows with the sampled queries
            self.df = self.df[self.df['query'].isin(sampled_queries)]
            
            print(f"Dataset reduced from {len(unique_queries)} unique queries to {len(sampled_queries)} queries")
            print(f"Total rows reduced from original size to {len(self.df)} rows")

        # Create a dictionary to map product locales to their respective relevance scores
        esci_dict = {"E":4,"S":3,"C":2,"I":1}
        self.df['esci_label'] = self.df['esci_label'].map(esci_dict)

        # Rename the columns for clarity
        self.df.rename(columns={
            'esci_label': 'relevance',
            'product_title': 'title',
            'product_description': 'description',
        }, inplace=True)

        # Drop na values in the 'revelance' column
        self.df = self.df.dropna(subset=['relevance','title','query'])

        if self.small_version:
            # Create a small version of the dataset for testing purposes
            self.df = self.df[self.df['small_version'] == 1]

        # Only keep the relevant columns
        self.df = self.df[['query', 'title', 'description', 
                           'relevance','split','product_locale']]