import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SalesDataAnalyzer:
    """
    A comprehensive sales data analysis and visualization tool
    """
    
    def __init__(self, file_path=None):
        """
        Constructor to initialize the SalesDataAnalyzer
        """
        self.data = None
        self.current_plot = None
        self.loaded = False
        
        if file_path:
            self.load_data(file_path)
    
    def __del__(self):
        """
        Destructor to perform cleanup
        """
        print("Cleaning up SalesDataAnalyzer resources...")
    
    def load_data(self, file_path):
        """
        Load data from CSV file
        """
        try:
            if not os.path.exists(file_path):
                # Create sample data if file doesn't exist
                print(f"File {file_path} not found. Creating sample data...")
                self._create_sample_data()
            else:
                self.data = pd.read_csv(file_path)
                self.loaded = True
                print(f"Data loaded successfully from {file_path}")
                print(f"Dataset shape: {self.data.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Create sample sales data for demonstration
        """
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Monitor']
        regions = ['North', 'South', 'East', 'West']
        
        sample_data = []
        for date in dates[:100]:  # Create 100 sample records
            product = np.random.choice(products)
            region = np.random.choice(regions)
            sales = np.random.randint(1000, 5000)
            profit = sales * np.random.uniform(0.1, 0.3)
            quantity = np.random.randint(1, 10)
            
            sample_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Product': product,
                'Region': region,
                'Sales': sales,
                'Profit': profit,
                'Quantity': quantity,
                'Year': date.year,
                'Month': date.month
            })
        
        self.data = pd.DataFrame(sample_data)
        self.loaded = True
        print("Sample data created successfully!")
    
    def explore_data(self):
        """
        Display basic information about the dataset
        """
        if not self._check_data_loaded():
            return
        
        print("=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)
        
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nDataset Description:")
        print(self.data.describe())
        
        print("\nColumn Names:")
        print(self.data.columns.tolist())
        
        print(f"\nDataset Shape: {self.data.shape}")
    
    def clean_data(self):
        """
        Handle missing values and perform data cleaning
        """
        if not self._check_data_loaded():
            return
        
        print("=" * 50)
        print("DATA CLEANING")
        print("=" * 50)
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        print("\nMissing values in each column:")
        print(missing_values)
        
        if missing_values.sum() == 0:
            print("No missing values found!")
            return
        
        # Handle missing values
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                if self.data[column].dtype in ['int64', 'float64']:
                    # Fill numerical columns with mean
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                    print(f"Filled missing values in {column} with mean")
                else:
                    # Fill categorical columns with mode
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                    print(f"Filled missing values in {column} with mode")
    
    def handle_missing_data_menu(self):
        """
        Menu for handling missing data
        """
        if not self._check_data_loaded():
            return
        
        while True:
            print("\n== Handle Missing Data ==")
            print("1. Display rows with missing values")
            print("2. Fill missing values with mean (numerical) or mode (categorical)")
            print("3. Drop rows with missing values")
            print("4. Replace missing values with specific value")
            print("5. Back to main menu")
            
            choice = input("Enter your choice: ").strip()
            
            if choice == '1':
                self._display_missing_values()
            elif choice == '2':
                self._fill_missing_values_auto()
            elif choice == '3':
                self._drop_missing_values()
            elif choice == '4':
                self._replace_missing_values()
            elif choice == '5':
                break
            else:
                print("Invalid choice! Please try again.")
    
    def _display_missing_values(self):
        """Display rows with missing values"""
        missing_rows = self.data[self.data.isnull().any(axis=1)]
        if missing_rows.empty:
            print("No missing values found in the dataset!")
        else:
            print(f"Found {len(missing_rows)} rows with missing values:")
            print(missing_rows)
    
    def _fill_missing_values_auto(self):
        """Fill missing values automatically"""
        self.clean_data()
    
    def _drop_missing_values(self):
        """Drop rows with missing values"""
        initial_shape = self.data.shape
        self.data.dropna(inplace=True)
        final_shape = self.data.shape
        rows_dropped = initial_shape[0] - final_shape[0]
        print(f"Dropped {rows_dropped} rows with missing values.")
    
    def _replace_missing_values(self):
        """Replace missing values with specific value"""
        column = input("Enter column name: ").strip()
        if column not in self.data.columns:
            print("Column not found!")
            return
        
        value = input("Enter value to replace missing values: ").strip()
        
        # Convert value to appropriate type
        if self.data[column].dtype in ['int64', 'float64']:
            try:
                value = float(value)
            except ValueError:
                print("Invalid numerical value!")
                return
        else:
            value = str(value)
        
        self.data[column].fillna(value, inplace=True)
        print(f"Replaced missing values in {column} with {value}")
    
    def mathematical_operations(self):
        """
        Perform mathematical operations on data
        """
        if not self._check_data_loaded():
            return
        
        print("=" * 50)
        print("MATHEMATICAL OPERATIONS")
        print("=" * 50)
        
        # Convert relevant columns to numpy arrays
        sales_array = self.data['Sales'].to_numpy()
        profit_array = self.data['Profit'].to_numpy()
        
        print(f"Sales array shape: {sales_array.shape}")
        print(f"Profit array shape: {profit_array.shape}")
        
        # Element-wise operations
        print("\nElement-wise operations:")
        print(f"Sales + Profit: {sales_array + profit_array}")
        print(f"Sales * 1.1 (10% increase): {sales_array * 1.1}")
        
        # Statistical operations
        print(f"\nSales Statistics:")
        print(f"Sum: {np.sum(sales_array)}")
        print(f"Mean: {np.mean(sales_array):.2f}")
        print(f"Standard Deviation: {np.std(sales_array):.2f}")
        
        # Array indexing and slicing
        print(f"\nArray Indexing and Slicing:")
        print(f"First 5 sales: {sales_array[:5]}")
        print(f"Last 5 sales: {sales_array[-5:]}")
        print(f"Sales at index 10: {sales_array[10]}")
    
    def combine_data(self, other_dataframe):
        """
        Combine current DataFrame with another
        """
        if not self._check_data_loaded():
            return None
        
        try:
            combined_data = pd.concat([self.data, other_dataframe], ignore_index=True)
            print(f"Data combined successfully! New shape: {combined_data.shape}")
            return combined_data
        except Exception as e:
            print(f"Error combining data: {e}")
            return None
    
    def split_data(self):
        """
        Split DataFrame into multiple DataFrames based on criteria
        """
        if not self._check_data_loaded():
            return {}
        
        print("=" * 50)
        print("DATA SPLITTING")
        print("=" * 50)
        
        split_data = {}
        
        # Split by region
        regions = self.data['Region'].unique()
        for region in regions:
            split_data[f'region_{region}'] = self.data[self.data['Region'] == region]
            print(f"Data for {region}: {split_data[f'region_{region}'].shape}")
        
        # Split by product
        products = self.data['Product'].unique()
        for product in products:
            split_data[f'product_{product}'] = self.data[self.data['Product'] == product]
            print(f"Data for {product}: {split_data[f'product_{product}'].shape}")
        
        return split_data
    
    def search_sort_filter(self):
        """
        Implement search, sort, and filter functionalities
        """
        if not self._check_data_loaded():
            return
        
        print("=" * 50)
        print("SEARCH, SORT AND FILTER")
        print("=" * 50)
        
        # Search for specific products
        product_to_search = input("Enter product name to search (or press enter to skip): ").strip()
        if product_to_search:
            result = self.data[self.data['Product'].str.contains(product_to_search, case=False, na=False)]
            print(f"\nSearch results for '{product_to_search}':")
            print(result if not result.empty else "No products found!")
        
        # Sort data
        print("\nSorting options:")
        print("1. Sort by Sales (Descending)")
        print("2. Sort by Profit (Descending)")
        print("3. Sort by Date (Ascending)")
        
        sort_choice = input("Enter sorting choice (1-3): ").strip()
        if sort_choice == '1':
            sorted_data = self.data.sort_values('Sales', ascending=False)
        elif sort_choice == '2':
            sorted_data = self.data.sort_values('Profit', ascending=False)
        elif sort_choice == '3':
            sorted_data = self.data.sort_values('Date', ascending=True)
        else:
            sorted_data = self.data
        
        print("\nSorted Data (first 5 rows):")
        print(sorted_data.head())
        
        # Filter data
        print("\nFiltering options:")
        min_sales = input("Enter minimum sales value to filter (or press enter to skip): ").strip()
        if min_sales:
            try:
                min_sales = float(min_sales)
                filtered_data = self.data[self.data['Sales'] >= min_sales]
                print(f"\nFiltered Data (Sales >= {min_sales}):")
                print(f"Records found: {len(filtered_data)}")
                print(filtered_data.head())
            except ValueError:
                print("Invalid sales value!")
    
    def aggregate_functions(self):
        """
        Apply aggregating functions like sum, mean, etc.
        """
        if not self._check_data_loaded():
            return
        
        print("=" * 50)
        print("AGGREGATE FUNCTIONS")
        print("=" * 50)
        
        # Group by Product
        product_agg = self.data.groupby('Product').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Profit': ['sum', 'mean'],
            'Quantity': 'sum'
        }).round(2)
        
        print("Aggregation by Product:")
        print(product_agg)
        
        # Group by Region
        region_agg = self.data.groupby('Region').agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean']
        }).round(2)
        
        print("\nAggregation by Region:")
        print(region_agg)
    
    def statistical_analysis(self):
        """
        Perform statistical computations
        """
        if not self._check_data_loaded():
            return
        
        print("=" * 50)
        print("STATISTICAL ANALYSIS")
        print("=" * 50)
        
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns
        
        for column in numerical_columns:
            print(f"\nStatistics for {column}:")
            print(f"Mean: {self.data[column].mean():.2f}")
            print(f"Median: {self.data[column].median():.2f}")
            print(f"Standard Deviation: {self.data[column].std():.2f}")
            print(f"Variance: {self.data[column].var():.2f}")
            print(f"Min: {self.data[column].min()}")
            print(f"Max: {self.data[column].max()}")
            print(f"25th Percentile: {self.data[column].quantile(0.25):.2f}")
            print(f"75th Percentile: {self.data[column].quantile(0.75):.2f}")
    
    def generate_descriptive_statistics(self):
        """
        Generate comprehensive descriptive statistics
        """
        if not self._check_data_loaded():
            return
        
        print("=" * 50)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 50)
        
        print(self.data.describe(include='all'))
    
    def create_pivot_table(self):
        """
        Generate pivot tables for data summarization
        """
        if not self._check_data_loaded():
            return
        
        print("=" * 50)
        print("PIVOT TABLES")
        print("=" * 50)
        
        # Pivot table: Sales by Product and Region
        pivot1 = pd.pivot_table(self.data, 
                               values='Sales', 
                               index='Product', 
                               columns='Region', 
                               aggfunc='sum',
                               fill_value=0)
        
        print("Pivot Table - Total Sales by Product and Region:")
        print(pivot1)
        
        # Pivot table: Average Profit by Month and Product
        if 'Month' in self.data.columns:
            pivot2 = pd.pivot_table(self.data,
                                   values='Profit',
                                   index='Month',
                                   columns='Product',
                                   aggfunc='mean',
                                   fill_value=0)
            
            print("\nPivot Table - Average Profit by Month and Product:")
            print(pivot2)
    
    def visualize_data(self):
        """
        Create various plots using Matplotlib and Seaborn
        """
        if not self._check_data_loaded():
            return
        
        while True:
            print("\n== Data Visualization ==")
            print("1. Bar Plot")
            print("2. Line Plot")
            print("3. Scatter Plot")
            print("4. Pie Chart")
            print("5. Histogram")
            print("6. Stack Plot")
            print("7. Box Plot (Seaborn)")
            print("8. Heatmap (Seaborn)")
            print("9. Back to main menu")
            
            choice = input("Enter your choice: ").strip()
            
            if choice == '1':
                self._create_bar_plot()
            elif choice == '2':
                self._create_line_plot()
            elif choice == '3':
                self._create_scatter_plot()
            elif choice == '4':
                self._create_pie_chart()
            elif choice == '5':
                self._create_histogram()
            elif choice == '6':
                self._create_stack_plot()
            elif choice == '7':
                self._create_box_plot()
            elif choice == '8':
                self._create_heatmap()
            elif choice == '9':
                break
            else:
                print("Invalid choice! Please try again.")
    
    def _create_bar_plot(self):
        """Create bar plot"""
        try:
            # Sales by Product
            sales_by_product = self.data.groupby('Product')['Sales'].sum()
            
            plt.figure(figsize=(10, 6))
            sales_by_product.plot(kind='bar', color='skyblue')
            plt.title('Total Sales by Product', fontsize=14, fontweight='bold')
            plt.xlabel('Product')
            plt.ylabel('Total Sales')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            self.current_plot = plt.gcf()
            print("Bar plot displayed successfully!")
        except Exception as e:
            print(f"Error creating bar plot: {e}")
    
    def _create_line_plot(self):
        """Create line plot"""
        try:
            if 'Month' in self.data.columns:
                monthly_sales = self.data.groupby('Month')['Sales'].sum()
                
                plt.figure(figsize=(10, 6))
                monthly_sales.plot(kind='line', marker='o', color='green', linewidth=2)
                plt.title('Monthly Sales Trend', fontsize=14, fontweight='bold')
                plt.xlabel('Month')
                plt.ylabel('Total Sales')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                self.current_plot = plt.gcf()
                print("Line plot displayed successfully!")
            else:
                print("Month column not available for line plot.")
        except Exception as e:
            print(f"Error creating line plot: {e}")
    
    def _create_scatter_plot(self):
        """Create scatter plot"""
        try:
            print("Available numerical columns:", self.data.select_dtypes(include=[np.number]).columns.tolist())
            
            x_col = input("Enter x-axis column name: ").strip()
            y_col = input("Enter y-axis column name: ").strip()
            
            if x_col not in self.data.columns or y_col not in self.data.columns:
                print("One or both columns not found!")
                return
            
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[x_col], self.data[y_col], alpha=0.6, color='red')
            plt.title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            self.current_plot = plt.gcf()
            print("Scatter plot displayed successfully!")
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
    
    def _create_pie_chart(self):
        """Create pie chart"""
        try:
            # Sales distribution by Region
            sales_by_region = self.data.groupby('Region')['Sales'].sum()
            
            plt.figure(figsize=(8, 8))
            plt.pie(sales_by_region, labels=sales_by_region.index, autopct='%1.1f%%', startangle=90)
            plt.title('Sales Distribution by Region', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            self.current_plot = plt.gcf()
            print("Pie chart displayed successfully!")
        except Exception as e:
            print(f"Error creating pie chart: {e}")
    
    def _create_histogram(self):
        """Create histogram"""
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(self.data['Sales'], bins=20, color='orange', alpha=0.7, edgecolor='black')
            plt.title('Distribution of Sales', fontsize=14, fontweight='bold')
            plt.xlabel('Sales')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            self.current_plot = plt.gcf()
            print("Histogram displayed successfully!")
        except Exception as e:
            print(f"Error creating histogram: {e}")
    
    def _create_stack_plot(self):
        """Create stack plot"""
        try:
            if 'Month' in self.data.columns and 'Product' in self.data.columns:
                # Prepare data for stack plot
                pivot_data = pd.pivot_table(self.data, 
                                          values='Sales', 
                                          index='Month', 
                                          columns='Product', 
                                          aggfunc='sum',
                                          fill_value=0)
                
                plt.figure(figsize=(12, 8))
                plt.stackplot(pivot_data.index, pivot_data.T, labels=pivot_data.columns)
                plt.title('Monthly Sales by Product (Stacked)', fontsize=14, fontweight='bold')
                plt.xlabel('Month')
                plt.ylabel('Sales')
                plt.legend(loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                self.current_plot = plt.gcf()
                print("Stack plot displayed successfully!")
            else:
                print("Required columns (Month, Product) not available for stack plot.")
        except Exception as e:
            print(f"Error creating stack plot: {e}")
    
    def _create_box_plot(self):
        """Create box plot using Seaborn"""
        try:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.data, x='Product', y='Sales')
            plt.title('Sales Distribution by Product', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            self.current_plot = plt.gcf()
            print("Box plot displayed successfully!")
        except Exception as e:
            print(f"Error creating box plot: {e}")
    
    def _create_heatmap(self):
        """Create heatmap using Seaborn"""
        try:
            # Correlation heatmap
            numerical_data = self.data.select_dtypes(include=[np.number])
            correlation_matrix = numerical_data.corr()
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            self.current_plot = plt.gcf()
            print("Heatmap displayed successfully!")
        except Exception as e:
            print(f"Error creating heatmap: {e}")
    
    def save_visualization(self):
        """
        Save the current visualization to a file
        """
        if self.current_plot is None:
            print("No plot to save! Please create a visualization first.")
            return
        
        filename = input("Enter file name to save the plot (e.g., plot.png): ").strip()
        if not filename:
            print("Invalid file name!")
            return
        
        try:
            self.current_plot.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {filename} successfully!")
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    def _check_data_loaded(self):
        """
        Check if data is loaded
        """
        if not self.loaded or self.data is None:
            print("No data loaded! Please load data first.")
            return False
        return True

def main():
    """
    Main function to run the Sales Data Analysis program
    """
    analyzer = SalesDataAnalyzer()
    
    print("=" * 60)
    print("      SALES DATA ANALYSIS & VISUALIZATION TOOL")
    print("=" * 60)
    
    while True:
        print("\n" + "=" * 60)
        print("Please select an option:")
        print("1. Load Dataset")
        print("2. Explore Data")
        print("3. Perform Mathematical Operations")
        print("4. Handle Missing Data")
        print("5. Generate Descriptive Statistics")
        print("6. Search, Sort and Filter Data")
        print("7. Aggregate Functions")
        print("8. Statistical Analysis")
        print("9. Create Pivot Tables")
        print("10. Data Visualization")
        print("11. Save Visualization")
        print("12. Split Data")
        print("13. Exit")
        print("-" * 60)
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            file_path = input("Enter CSV file path (or press enter for sample data): ").strip()
            if not file_path:
                file_path = 'sales_data.csv'
            analyzer.load_data(file_path)
        
        elif choice == '2':
            analyzer.explore_data()
        
        elif choice == '3':
            analyzer.mathematical_operations()
        
        elif choice == '4':
            analyzer.handle_missing_data_menu()
        
        elif choice == '5':
            analyzer.generate_descriptive_statistics()
        
        elif choice == '6':
            analyzer.search_sort_filter()
        
        elif choice == '7':
            analyzer.aggregate_functions()
        
        elif choice == '8':
            analyzer.statistical_analysis()
        
        elif choice == '9':
            analyzer.create_pivot_table()
        
        elif choice == '10':
            analyzer.visualize_data()
        
        elif choice == '11':
            analyzer.save_visualization()
        
        elif choice == '12':
            analyzer.split_data()
        
        elif choice == '13':
            print("Exiting the program. Goodbye!")
            break
        
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(f"Required package not found: {e}")
        print("Please install required packages using:")
        print("pip install pandas matplotlib seaborn")
        exit(1)
    
    main()
