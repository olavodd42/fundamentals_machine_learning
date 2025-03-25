# Comprehensive Data Analysis for Familiar Blood Transfusion Startup

# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency

def load_data():
    """
    Load the datasets for lifespans and iron levels.
    
    Returns:
    tuple: DataFrames for lifespans and iron data
    """
    try:
        lifespans = pd.read_csv('familiar_lifespan.csv')
        iron = pd.read_csv('familiar_iron.csv')
        return lifespans, iron
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def analyze_vein_pack_lifespan(lifespans):
    """
    Analyze the lifespan of Vein Pack subscribers.
    
    Args:
    lifespans (pd.DataFrame): DataFrame containing lifespan data
    
    Returns:
    tuple: Mean lifespan and p-value of statistical test
    """
    # Extract lifespans for Vein Pack subscribers
    vein_pack_lifespans = lifespans[lifespans.pack == 'vein'].lifespan
    
    # Calculate mean lifespan
    mean_lifespan = np.mean(vein_pack_lifespans)
    print(f'Mean lifespan for Vein Pack subscribers: {mean_lifespan:.2f} years')
    
    # Perform one-sample t-test against 73 years
    t_statistic, p_value = ttest_1samp(vein_pack_lifespans, 73)
    print(f'P-value for Vein Pack lifespan test: {p_value:.4f}')
    
    return mean_lifespan, p_value

def compare_pack_lifespans(lifespans):
    """
    Compare lifespans between Vein and Artery Pack subscribers.
    
    Args:
    lifespans (pd.DataFrame): DataFrame containing lifespan data
    
    Returns:
    tuple: Mean lifespans and p-value of statistical test
    """
    # Extract lifespans for Vein and Artery Pack subscribers
    vein_pack_lifespans = lifespans[lifespans.pack == 'vein'].lifespan
    artery_pack_lifespans = lifespans[lifespans.pack == 'artery'].lifespan
    
    # Calculate mean lifespans
    vein_mean = np.mean(vein_pack_lifespans)
    artery_mean = np.mean(artery_pack_lifespans)
    print(f'Mean lifespan for Vein Pack subscribers: {vein_mean:.2f} years')
    print(f'Mean lifespan for Artery Pack subscribers: {artery_mean:.2f} years')
    
    # Perform two-sample t-test to compare lifespans
    t_statistic, p_value = ttest_ind(vein_pack_lifespans, artery_pack_lifespans)
    print(f'P-value for pack lifespan comparison: {p_value:.4f}')
    
    return vein_mean, artery_mean, p_value

def analyze_iron_levels(iron):
    """
    Analyze the association between pack type and iron levels.
    
    Args:
    iron (pd.DataFrame): DataFrame containing iron level data
    
    Returns:
    tuple: Contingency table and p-value of chi-square test
    """
    # Create contingency table
    contingency_table = pd.crosstab(iron.pack, iron.iron)
    print("Contingency Table of Pack and Iron Levels:")
    print(contingency_table)
    
    # Perform chi-square test of independence
    chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f'P-value for iron level association test: {p_value:.4f}')
    
    return contingency_table, p_value

def main():
    """
    Main function to orchestrate the entire analysis process.
    """
    # Load data
    lifespans, iron = load_data()
    if lifespans is None or iron is None:
        return
    
    # Display initial data
    print("\n--- Initial Lifespans Data ---")
    print(lifespans.head())
    
    print("\n--- Initial Iron Levels Data ---")
    print(iron.head())
    
    # Perform analysis
    print("\n--- Vein Pack Lifespan Analysis ---")
    vein_mean, vein_p_value = analyze_vein_pack_lifespan(lifespans)
    
    print("\n--- Pack Lifespan Comparison ---")
    vein_mean, artery_mean, comparison_p_value = compare_pack_lifespans(lifespans)
    
    print("\n--- Iron Levels Analysis ---")
    iron_table, iron_p_value = analyze_iron_levels(iron)
    
    # Interpretation of results
    print("\n--- Analysis Conclusions ---")
    
    # Vein Pack Lifespan Conclusion
    if vein_p_value < 0.05:
        print("The Vein Pack appears to have a statistically significant impact on lifespan.")
    else:
        print("There is no statistically significant evidence that the Vein Pack affects lifespan.")
    
    # Pack Comparison Conclusion
    if comparison_p_value < 0.05:
        print("There is a statistically significant difference in lifespans between Vein and Artery Packs.")
    else:
        print("There is no statistically significant difference in lifespans between Vein and Artery Packs.")
    
    # Iron Levels Conclusion
    if iron_p_value < 0.05:
        print("There is a statistically significant association between pack type and iron levels.")
    else:
        print("There is no statistically significant association between pack type and iron levels.")

# Run the analysis
if __name__ == "__main__":
    main()