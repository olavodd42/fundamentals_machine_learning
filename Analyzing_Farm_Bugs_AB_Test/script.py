# Farmburg A/B Test Analysis
# This script analyzes the effectiveness of different pricing strategies 
# for a microtransaction upgrade in a farming simulation game

# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, binom_test

def load_and_analyze_data(file_path='clicks.csv'):
    """
    Load the A/B test data and perform initial analysis.
    This includes creating a contingency table and performing a Chi-Square test.
    
    Args:
        file_path (str): Path to the CSV file containing click data.
    
    Returns:
        tuple: Containing DataFrame, contingency table, and chi-square p-value.
    """
    
    # Load the data
    try:
      abdata = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None, None
    
    # Print initial data overview
    print("Initial Data Overview:")
    print(abdata.head())
    
    # Create contingency table of group and purchase
    contingency_table = pd.crosstab(abdata.group, abdata.is_purchase)
    print("\nContingency Table:")
    print(contingency_table)
    
    # Perform Chi-Square test to determine if there is a significant association between group and purchase
    chi2_stat, p_value, *_ = chi2_contingency(contingency_table)
    print(f"\nChi-Square Test P-value: {p_value:.4f}")
    
    return abdata, contingency_table, p_value

def calculate_revenue_targets(num_visits, price_points):
    """
    Calculate sales and purchase percentages needed to meet revenue target
    
    Args:
        num_visits (int): Number of weekly site visitors
        price_points (list): List of different price points
    
    Returns:
        dict: Sales needed and purchase percentages for each price point
    """
    # Minimum weekly revenue target
    REVENUE_TARGET = 1000
    
    revenue_analysis = {}
    for price in price_points:
        # Calculate number of sales needed to reach revenue target
        num_sales_needed = np.ceil(REVENUE_TARGET / price)
        
        # Calculate percentage of visitors needed to make a purchase
        p_sales_needed = (num_sales_needed / num_visits) * 100
        
        revenue_analysis[price] = {
            'num_sales_needed': num_sales_needed,
            'p_sales_needed': p_sales_needed
        }
    
    # Print detailed revenue analysis
    print("\nRevenue Target Analysis:")
    for price, details in revenue_analysis.items():
        print(f"For ${price:.2f} price point:")
        print(f"  - Total weekly visitors: {num_visits}")
        print(f"  - Sales needed to reach ${REVENUE_TARGET}: {details['num_sales_needed']}")
        print(f"  - Percentage of visitors needed to purchase: {details['p_sales_needed']:.2f}%")
    
    return revenue_analysis

def perform_binomial_tests(contingency_table, revenue_analysis):
    """
    Perform binomial tests for each price point group
    
    Args:
        contingency_table (pd.DataFrame): Contingency table of purchases
        revenue_analysis (dict): Sales and percentage targets for each price point
    
    Returns:
        dict: P-values for each group's binomial test
    """
    # Price points and their corresponding group labels
    price_groups = dict(zip([0.99, 1.99, 4.99], ['A', 'B', 'C']))
    
    binomial_results = {}
    
    print("\nBinomial Test Results:")
    for price, group in price_groups.items():
        # Get sample size and number of sales for the group
        sample_size = contingency_table.loc[group].sum()
        sales = contingency_table.loc[group, 'Yes']
        
        # Get the required purchase percentage for the price point
        p_sales_needed = revenue_analysis[price]['p_sales_needed'] / 100
        
        # Perform binomial test
        p_value = binom_test(
            sales, 
            sample_size, 
            p=p_sales_needed, 
            alternative='greater'
        )
        
        binomial_results[group] = {
            'price': price,
            'sample_size': sample_size,
            'sales': sales,
            'p_value': p_value
        }
        
        print(f"Group {group} (${price:.2f}):")
        print(f"  - Sample Size: {sample_size}")
        print(f"  - Sales: {sales}")
        print(f"  - Binomial Test P-value: {p_value:.4f}")
    
    return binomial_results

def main():
    """
    Main function to orchestrate the A/B test analysis
    """
    # Price points to analyze
    PRICE_POINTS = [0.99, 1.99, 4.99]
    
    # Load and analyze initial data
    abdata, contingency_table, chi2_p_value = load_and_analyze_data()
    
    # Calculate number of weekly visits
    num_visits = len(abdata)
    
    # Calculate revenue targets
    revenue_analysis = calculate_revenue_targets(num_visits, PRICE_POINTS)
    
    # Perform binomial tests
    binomial_results = perform_binomial_tests(contingency_table, revenue_analysis)
    
    # Determine recommended price point
    print("\nRecommendation:")
    recommended_groups = [
        group for group, result in binomial_results.items() 
        if result['p_value'] < 0.05
    ]
    
    if recommended_groups:
        recommended_price = binomial_results[recommended_groups[0]]['price']
        print(f"Recommended price point: ${recommended_price:.2f}")
        print("This price point can statistically meet the minimum revenue target.")
    else:
        print("No price point statistically meets the minimum revenue target.")

# Run the analysis
if __name__ == "__main__":
    main()