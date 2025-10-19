import pandas as pd

# Load dataset safely
df = pd.read_csv("complaints.csv", on_bad_lines='skip', engine='python')

# Print unique product names
print("\n✅ Unique Product names in your dataset:\n")
print(df["Product"].unique())

