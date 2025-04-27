import pandas as pd

class DataLoader:
    def __init__(self, example_path,products_path,small_version=True):
        """
        Initialize the DataLoader with paths to the datasets."""
        self.example_path = example_path
        self.products_path = products_path
        self.small_version = small_version
        self.df = None

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