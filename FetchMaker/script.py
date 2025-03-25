# Comprehensive Dog Breed Analysis for FetchMaker
# This script performs statistical analysis on dog breed data to identify significant differences in rescue rates, weights, and color associations.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest, f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class FetchMakerAnalyzer:
    """
    A comprehensive analysis tool for FetchMaker dog breed data
    """
    def __init__(self, data_path='dog_data.csv'):
        """
        Initialize the analyzer with dog data
        
        Args:
        data_path (str): Path to the CSV file containing dog data
        """
        try:
            self.dogs = pd.read_csv(data_path)
            print("Data successfully loaded.")
        except pd.errors.EmptyDataError:
            print("Error: The file is empty.")
            self.dogs = None
        except FileNotFoundError:
            print(f"Error: File {data_path} not found.")
            self.dogs = None

    def analyze_rescue_proportion(self, breed='whippet', expected_rescue_rate=0.08):
        """
        Analyze the proportion of rescue dogs for a specific breed
        
        Args:
        breed (str): Breed to analyze
        expected_rescue_rate (float): Expected proportion of rescue dogs
        
        Returns:
        dict: Analysis results including rescue count, total count, and p-value
        """
        if self.dogs is None:
            return None
        
        breed_data = self.dogs[self.dogs.breed == breed]
        rescue_data = breed_data.is_rescue
        
        num_rescues = np.sum(rescue_data)
        total_count = len(rescue_data)
        
        # Binomial test to check if rescue proportion differs from expected
        p_value = binomtest(num_rescues, total_count, p=expected_rescue_rate).pvalue
        
        print(f"\n--- {breed.capitalize()} Rescue Analysis ---")
        print(f"Total {breed}s: {total_count}")
        print(f"Number of {breed} rescues: {num_rescues}")
        print(f"Rescue proportion: {num_rescues/total_count:.2%}")
        print(f"P-value: {p_value:.4f}")
        
        return {
            'total_count': total_count,
            'rescue_count': num_rescues,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }

    def compare_breed_weights(self, breeds=['whippet', 'terrier', 'pitbull']):
        """
        Compare weights across specified dog breeds
        
        Args:
        breeds (list): List of breeds to compare
        
        Returns:
        dict: Analysis results including ANOVA p-value and pairwise comparisons
        """
        if self.dogs is None:
            return None
        
        # Subset data to specified breeds
        dogs_subset = self.dogs[self.dogs.breed.isin(breeds)]
        
        # Separate weight data for each breed
        breed_weights = [dogs_subset[dogs_subset.breed == breed].weight for breed in breeds]
        
        # Perform one-way ANOVA
        f_statistic, anova_p_value = f_oneway(*breed_weights)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='breed', y='weight', data=dogs_subset)
        plt.title(f'Weight Comparison: {", ".join(breeds)} Breeds')
        plt.xlabel('Breed')
        plt.ylabel('Weight (lbs)')
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Tukey's HSD for pairwise comparisons
        tukey_results = pairwise_tukeyhsd(
            endog=dogs_subset.weight, 
            groups=dogs_subset.breed, 
            alpha=0.05
        )
        
        print("\n--- Breed Weight Comparison ---")
        print(f"ANOVA p-value: {anova_p_value:.4f}")
        print("\nTukey HSD Pairwise Comparisons:")
        print(tukey_results)
        
        return {
            'anova_p_value': anova_p_value,
            'is_significant': anova_p_value < 0.05,
            'tukey_results': tukey_results
        }

    def analyze_breed_color_association(self, breeds=['poodle', 'shihtzu']):
        """
        Analyze color distribution across specified breeds
        
        Args:
        breeds (list): List of breeds to compare colors
        
        Returns:
        dict: Analysis results including contingency table and chi-square results
        """
        if self.dogs is None:
            return None
        
        # Subset data to specified breeds
        dogs_subset = self.dogs[self.dogs.breed.isin(breeds)]
        
        # Create contingency table
        contingency_table = pd.crosstab(dogs_subset.color, dogs_subset.breed)
        
        # Perform chi-square test
        chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        contingency_table.plot(kind='bar', stacked=True)
        plt.title(f'Color Distribution: {" vs ".join(breeds)}')
        plt.xlabel('Color')
        plt.ylabel('Count')
        plt.legend(title='Breed', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        plt.close()
        
        print("\n--- Breed Color Association ---")
        print("Contingency Table:")
        print(contingency_table)
        print(f"\nChi-square p-value: {p_value:.4f}")
        
        return {
            'contingency_table': contingency_table,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }

    def comprehensive_analysis(self):
        """
        Perform a comprehensive analysis of the dog breed data
        """
        if self.dogs is None:
            print("Cannot perform analysis. Data not loaded.")
            return
        
        print("\n=== FetchMaker Dog Data Analysis ===")
        
        # Initial data overview
        print("\nData Overview:")
        print(self.dogs.head())
        print("\nBreed Distribution:")
        print(self.dogs.breed.value_counts())
        
        # Rescue proportion analysis for whippets
        self.analyze_rescue_proportion()
        
        # Weight comparison for mid-sized breeds
        self.compare_breed_weights()
        
        # Color association for poodles and shihtzus
        self.analyze_breed_color_association()

def main():
    """
    Main function to run the FetchMaker data analysis
    """
    analyzer = FetchMakerAnalyzer()
    analyzer.comprehensive_analysis()

if __name__ == "__main__":
    main()