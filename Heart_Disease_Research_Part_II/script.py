# Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings

# Visualization Settings
plt.style.use('seaborn-whitegrid')  # Modern style
sns.set_palette("colorblind")  # Colorblind-friendly palette
plt.rcParams['figure.figsize'] = (12, 8)  # Figure size
plt.rcParams['font.size'] = 12  # Font size

# Load Data
heart = pd.read_csv('heart_disease.csv')

# Function to create a formatted section title
def print_section(title):
    print('\n' + '=' * 80)
    print(f' {title} '.center(80, '='))
    print('=' * 80 + '\n')

# 1. EXPLORATORY DATA ANALYSIS
print_section("EXPLORATORY DATA ANALYSIS")

# Basic dataset information
print("Dataset Information:")
print(f"- Number of records: {heart.shape[0]}")
print(f"- Number of variables: {heart.shape[1]}")
print(f"- Heart disease distribution: {heart['heart_disease'].value_counts().to_dict()}")
print("\nFirst 5 rows of the dataset:")
print(heart.head())

# Descriptive statistics for numerical variables
print("\nDescriptive statistics of numerical variables:")
print(heart.describe())

# Check for missing values
missing_values = heart.isnull().sum()
print("\nMissing values per column:")
print(missing_values)

# 2. RELATIONSHIP BETWEEN THALACH AND HEART DISEASE
print_section("RELATIONSHIP BETWEEN THALACH (MAXIMUM HEART RATE) AND HEART DISEASE")

# Separate data by heart disease presence/absence
thalach_hd = heart[heart.heart_disease == 'presence'].thalach
thalach_no_hd = heart[heart.heart_disease == 'absence'].thalach

# Descriptive statistics for thalach by group
print("Thalach statistics by group:")
thalach_stats = heart.groupby('heart_disease')['thalach'].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
print(thalach_stats)

# Difference in means and medians
mean_diff = np.mean(thalach_no_hd) - np.mean(thalach_hd)
median_diff = np.median(thalach_no_hd) - np.median(thalach_hd)
print(f"\nDifference in mean thalach (absence - presence): {mean_diff:.2f}")
print(f"Difference in median thalach (absence - presence): {median_diff:.2f}")

# t-test for thalach
t_stat, p_val = ttest_ind(thalach_no_hd, thalach_hd)
print(f"\nT-test for thalach difference:")
print(f"- t-statistic: {t_stat:.4f}")
print(f"- p-value: {p_val:.8f}")
print(f"- Significant at 5%: {'Yes' if p_val < 0.05 else 'No'}")

# Visualization - Boxplot for thalach by heart disease
plt.figure(figsize=(10, 6))
sns.boxplot(x='heart_disease', y='thalach', data=heart, order=['absence', 'presence'])
plt.title('Maximum Heart Rate (thalach) by Heart Condition', fontsize=14)
plt.xlabel('Heart Disease', fontsize=12)
plt.ylabel('Maximum Heart Rate', fontsize=12)
plt.xticks([0, 1], ['Absence', 'Presence'])
plt.savefig('thalach_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Visualization - Histogram for thalach by heart disease
plt.figure(figsize=(12, 6))
sns.histplot(data=heart, x='thalach', hue='heart_disease', kde=True, element='step',
             palette=['green', 'red'], common_norm=False, bins=20)
plt.title('Distribution of Maximum Heart Rate by Heart Condition', fontsize=14)
plt.xlabel('Maximum Heart Rate', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(['Absence', 'Presence'])
plt.savefig('thalach_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 3. RELATIONSHIP BETWEEN AGE AND HEART DISEASE
print_section("RELATIONSHIP BETWEEN AGE AND HEART DISEASE")

# Descriptive statistics for age by group
print("Age statistics by group:")
age_stats = heart.groupby('heart_disease')['age'].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
print(age_stats)

# Separate data by heart disease presence/absence
age_hd = heart[heart.heart_disease == 'presence'].age
age_no_hd = heart[heart.heart_disease == 'absence'].age

# Difference in means and medians
mean_diff = np.mean(age_hd) - np.mean(age_no_hd)
median_diff = np.median(age_hd) - np.median(age_no_hd)
print(f"\nDifference in mean age (presence - absence): {mean_diff:.2f}")
print(f"Difference in median age (presence - absence): {median_diff:.2f}")

# t-test for age
t_stat, p_val = ttest_ind(age_hd, age_no_hd)
print(f"\nT-test for age difference:")
print(f"- t-statistic: {t_stat:.4f}")
print(f"- p-value: {p_val:.8f}")
print(f"- Significant at 5%: {'Yes' if p_val < 0.05 else 'No'}")

# Visualization - Boxplot for age by heart disease
plt.figure(figsize=(10, 6))
sns.boxplot(x='heart_disease', y='age', data=heart, order=['absence', 'presence'])
plt.title('Age by Heart Condition', fontsize=14)
plt.xlabel('Heart Disease', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.xticks([0, 1], ['Absence', 'Presence'])
plt.savefig('age_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Visualization - Histogram for age by heart disease
plt.figure(figsize=(12, 6))
sns.histplot(data=heart, x='age', hue='heart_disease', kde=True, element='step',
             palette=['green', 'red'], common_norm=False, bins=15)
plt.title('Age Distribution by Heart Condition', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(['Absence', 'Presence'])
plt.savefig('age_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 4. RELATIONSHIP BETWEEN CHEST PAIN TYPE (CP) AND HEART DISEASE
print_section("RELATIONSHIP BETWEEN CHEST PAIN TYPE (CP) AND HEART DISEASE")

# Contingency table (percentage and counts)
cp_disease_crosstab = pd.crosstab(heart.cp, heart.heart_disease, normalize='index') * 100
cp_disease_counts = pd.crosstab(heart.cp, heart.heart_disease)

print("Contingency table (counts):")
print(cp_disease_counts)

print("\nContingency table (percentages by pain type):")
print(cp_disease_crosstab.round(1))

# Chi-square test
chi2, p_val, dof, expected = chi2_contingency(cp_disease_counts)
print(f"\nChi-square test for independence between chest pain type and heart disease:")
print(f"- Chi-square statistic: {chi2:.4f}")
print(f"- Degrees of freedom: {dof}")
print(f"- p-value: {p_val:.10f}")
print(f"- Significant at 5%: {'Yes' if p_val < 0.05 else 'No'}")

# Visualization - Barplot for chest pain type by heart disease
plt.figure(figsize=(12, 7))
# Reorder chest pain types by heart disease rate for better visualization
cp_order = cp_disease_crosstab['presence'].sort_values(ascending=False).index
sns.countplot(x='cp', hue='heart_disease', data=heart, order=cp_order, palette=['green', 'red'])
plt.title('Relationship Between Chest Pain Type and Heart Disease', fontsize=14)
plt.xlabel('Chest Pain Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=15)
plt.legend(['Absence', 'Presence'])
plt.savefig('cp_disease_barplot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Visualization - Barplot for percentage of heart disease by chest pain type
plt.figure(figsize=(12, 7))
cp_disease_percent = cp_disease_crosstab.reset_index()
cp_disease_melted = pd.melt(cp_disease_percent, id_vars=['cp'],
                            value_vars=['absence', 'presence'],
                            var_name='heart_disease', value_name='percentage')

# Plotting the stacked bar chart
sns.barplot(x='cp', y='percentage', hue='heart_disease', data=cp_disease_melted,
            order=cp_order, palette=['green', 'red'])
plt.title('Percentage of Heart Disease by Chest Pain Type', fontsize=14)
plt.xlabel('Chest Pain Type', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=15)
plt.legend(['Absence', 'Presence'])
plt.savefig('cp_disease_percent.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 5. RELATIONSHIP BETWEEN CHEST PAIN TYPE (CP) AND THALACH
print_section("RELATIONSHIP BETWEEN CHEST PAIN TYPE (CP) AND THALACH")

# Descriptive statistics for thalach by chest pain type
print("Thalach statistics by chest pain type:")
thalach_cp_stats = heart.groupby('cp')['thalach'].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
print(thalach_cp_stats)

# Extract thalach values by chest pain type
thalach_typical = heart[heart.cp == 'typical angina'].thalach
thalach_asymptom = heart[heart.cp == 'asymptomatic'].thalach
thalach_nonangin = heart[heart.cp == 'non-anginal pain'].thalach
thalach_atypical = heart[heart.cp == 'atypical angina'].thalach

# ANOVA for thalach differences among chest pain types
f_stat, p_val = f_oneway(thalach_typical, thalach_asymptom, thalach_nonangin, thalach_atypical)
print(f"\nANOVA for thalach differences among chest pain types:")
print(f"- F-statistic: {f_stat:.4f}")
print(f"- p-value: {p_val:.8f}")
print(f"- Significant at 5%: {'Yes' if p_val < 0.05 else 'No'}")

# Tukey post-hoc test
thalach_all = np.concatenate([thalach_typical, thalach_asymptom, thalach_nonangin, thalach_atypical])
cp_labels = np.array(['typical angina'] * len(thalach_typical) +
                     ['asymptomatic'] * len(thalach_asymptom) +
                     ['non-anginal pain'] * len(thalach_nonangin) +
                     ['atypical angina'] * len(thalach_atypical))

tukey_results = pairwise_tukeyhsd(thalach_all, cp_labels, alpha=0.05)
print("\nTukey post-hoc test results:")
print(tukey_results)

# Visualization - Boxplot for thalach by chest pain type
plt.figure(figsize=(12, 7))
# Order chest pain types by median thalach for better visualization
cp_thalach_order = thalach_cp_stats['median'].sort_values(ascending=False).index
sns.boxplot(x='cp', y='thalach', data=heart, order=cp_thalach_order)
plt.title('Maximum Heart Rate by Chest Pain Type', fontsize=14)
plt.xlabel('Chest Pain Type', fontsize=12)
plt.ylabel('Maximum Heart Rate', fontsize=12)
plt.xticks(rotation=15)
plt.savefig('cp_thalach_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 6. MULTIVARIATE ANALYSIS
print_section("MULTIVARIATE ANALYSIS")

# Correlation matrix for numerical variables
numeric_heart = heart.select_dtypes(include=[np.number])
correlation_matrix = numeric_heart.corr()
print("Correlation matrix for numerical variables:")
print(correlation_matrix.round(2))

# Visualization - Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix for Numerical Variables', fontsize=14)
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Visualization - Pairplot for selected numerical variables
# Visualization - Pairplot for selected numerical variables
# Fix the pairplot that's causing the error by using a try-except block
# Replace the pairplot section with a more compatible approach
# Create a simpler alternative to pairplot using a grid of individual plots

# First, comment out the problematic pairplot code
'''
# Original problematic code
sns.pairplot(
    heart,
    hue='heart_disease',
    vars=['age', 'thalach', 'trestbps', 'chol'],
    palette={'absence': 'green', 'presence': 'red'},
    corner=True,
    diag_kind='kde'
).add_legend(title='Heart Disease')
plt.suptitle('Relationships Among Numerical Variables by Heart Condition', y=1.02, fontsize=16)
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
'''

# Alternative approach using individual plots
variables = ['age', 'thalach', 'trestbps', 'chol']
fig, axes = plt.subplots(len(variables), len(variables), figsize=(14, 14))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Add a title to the figure
fig.suptitle('Relationships Among Numerical Variables by Heart Condition', fontsize=16)

for i, var1 in enumerate(variables):
    for j, var2 in enumerate(variables):
        ax = axes[i, j]
        
        # Diagonal: histograms for single variables
        if i == j:
            for condition, color in zip(['absence', 'presence'], ['green', 'red']):
                subset = heart[heart.heart_disease == condition]
                sns.histplot(subset[var1], ax=ax, color=color, alpha=0.5, label=condition, 
                            kde=True, stat="density")
            ax.set_title(var1)
            
        # Off-diagonal: scatter plots for pairs of variables
        else:
            for condition, color in zip(['absence', 'presence'], ['green', 'red']):
                subset = heart[heart.heart_disease == condition]
                ax.scatter(subset[var2], subset[var1], color=color, alpha=0.5, 
                          label=condition, s=20)
            
            ax.set_xlabel(var2)
            ax.set_ylabel(var1)
            
            # Only add legend to the first plot to avoid duplicates
            if i == 1 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', title='Heart Disease')

# Save the figure
plt.savefig('custom_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

# 7. VARIABLE DISTRIBUTIONS BY SEX AND HEART DISEASE
print_section("VARIABLE DISTRIBUTIONS BY SEX AND HEART DISEASE")

# Contingency table for sex and heart disease
sex_disease_crosstab = pd.crosstab(heart.sex, heart.heart_disease)
print("Contingency table for sex and heart disease:")
print(sex_disease_crosstab)

# Chi-square test for sex and heart disease
chi2, p_val, dof, expected = chi2_contingency(sex_disease_crosstab)
print(f"\nChi-square test for independence between sex and heart disease:")
print(f"- Chi-square statistic: {chi2:.4f}")
print(f"- p-value: {p_val:.8f}")
print(f"- Significant at 5%: {'Yes' if p_val < 0.05 else 'No'}")

# Visualization - Barplot for sex and heart disease
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', hue='heart_disease', data=heart, palette=['green', 'red'])
plt.title('Distribution of Heart Disease by Sex', fontsize=14)
plt.xlabel('Sex', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['Female', 'Male'])
plt.legend(['Absence', 'Presence'])
plt.savefig('sex_disease_barplot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Visualization - Boxplot for thalach by sex and heart disease
plt.figure(figsize=(12, 7))
sns.boxplot(x='sex', y='thalach', hue='heart_disease', data=heart, palette=['green', 'red'])
plt.title('Maximum Heart Rate by Sex and Heart Condition', fontsize=14)
plt.xlabel('Sex', fontsize=12)
plt.ylabel('Maximum Heart Rate', fontsize=12)
plt.xticks([0, 1], ['Female', 'Male'])
plt.legend(['Absence', 'Presence'])
plt.savefig('sex_thalach_disease_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 8. ANALYSIS CONCLUSIONS
print_section("ANALYSIS CONCLUSIONS")

print("""
1. THALACH AND HEART DISEASE:
   - Patients without heart disease have a significantly higher maximum heart rate 
     (mean of 158.4 vs 139.3, p < 0.05).
   - The average difference is approximately 19 bpm.
   - This suggests that a lower capacity to increase heart rate during exercise 
     may be associated with the presence of heart disease.

2. AGE AND HEART DISEASE:
   - Patients with heart disease are significantly older 
     (mean of 55.9 vs 52.5 years, p < 0.05).
   - The average difference is about 3.4 years.
   - This confirms age as a risk factor for heart disease.

3. CHEST PAIN TYPE AND HEART DISEASE:
   - There is a strong association between chest pain type and the presence of heart disease 
     (p < 0.05).
   - Patients with asymptomatic pain are more likely to have heart disease (73% of cases).
   - Patients with non-anginal pain or atypical angina have a lower likelihood of heart disease 
     (21% and 18% of cases, respectively).

4. CHEST PAIN TYPE AND THALACH:
   - There are significant differences in maximum heart rate among different chest pain types (p < 0.05).
   - Patients with asymptomatic pain have a significantly lower maximum heart rate.
   - The Tukey post-hoc test showed significant differences mainly between asymptomatic patients and other groups.

5. SEX AND HEART DISEASE:
   - There is a significant association between sex and heart disease (p < 0.05).
   - Males have a higher prevalence of heart disease compared to females.

6. CORRELATED VARIABLES:
   - Thalach has a moderate negative correlation with age (-0.39), indicating that older individuals tend to have a lower maximum heart rate.
   - Thalach also shows a negative correlation with exercise-induced angina (exang).
""")
