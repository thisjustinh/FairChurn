import pandas as pd

# Load dataset from file, selecting only the desired columns
df = pd.read_csv("./data/churn.csv", usecols=['CreditScore', 'Age', 'Tenure', 'Balance', 'IsActiveMember', 'EstimatedSalary', 'Exited'])

# Select only numeric columns
num_cols = df.select_dtypes(include=['int', 'float']).columns

# Calculate statistics
mean = df[num_cols].mean() # Mean
median = df[num_cols].median() # Median
mode = df[num_cols].mode().iloc[0] # Mode
variance = df[num_cols].var() # Variance
std_dev = df[num_cols].std() # Standard deviation
minimum = df[num_cols].min() # Minimum
maximum = df[num_cols].max() # Maximum
range = maximum - minimum # Range
max_likelihood = df[num_cols].max() # Maximum likelihood

# Print statistics
print("The Mean:\n", mean,
      "\nThe Median:\n", median,
      "\nThe Mode:\n", mode,
      "\nThe Variance:\n", variance,
      "\nThe Standard Deviation:\n", std_dev,
      "\nThe Minimum:\n", minimum,
      "\nThe Maximum:\n", maximum,
      "\nThe Range:\n", range,
      "\nThe Maximum Likelihood:\n", max_likelihood)
