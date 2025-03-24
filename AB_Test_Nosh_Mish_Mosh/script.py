import noshmishmosh
import numpy as np

# Step 1-2: Load visitor and purchase data
all_visitors = noshmishmosh.customer_visits
paying_visitors = noshmishmosh.purchasing_customers

# Step 3-4: Calculate total and paying visitor counts
total_visitor_count = len(all_visitors)
paying_visitor_count = len(paying_visitors)

# Step 5: Calculate the baseline conversion percentage by dividing the number of paying visitors by the total number of visitors and multiplying by 100 to get a percentage.
baseline_percent = (paying_visitor_count / total_visitor_count) * 100.0 if total_visitor_count != 0 else 0
print(f'Baseline percent of paying visitors: {baseline_percent:.2f}%')

# Step 6-7: Calculate average payment per customer
payment_history = noshmishmosh.money_spent
average_payment = np.mean(payment_history)

# Step 8: Calculate the number of new customers needed to achieve an additional $1240 in revenue based on the average payment per customer.
new_customers_needed = np.ceil(1240 / average_payment)

# Step 9: Calculate percent increase needed in paying visitors
def calculate_percentage_point_increase(new_customers, total_visitors):
    return (new_customers / total_visitors) * 100

percentage_point_increase = calculate_percentage_point_increase(new_customers_needed, total_visitor_count)
print(f'Percentage point increase needed: {percentage_point_increase:.2f}%')

# Step 10: Minimum detectable effect (MDE)
mde = (percentage_point_increase / baseline_percent) * 100.0
print(f'Minimum detectable effect: {mde:.2f}%')

# Step 11: Define the statistical significance threshold. For this project, we assume it to be 10%.
# Step 12: Based on external calculator, store final sample size required
ab_sample_size = 490
print(f'Suggested A/B test sample size per group: {ab_sample_size}')
