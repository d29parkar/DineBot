import pandas as pd

# Load the CSV file
file_path = "menudata_internal_data.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Remove unnecessary columns
df = df.drop(columns=["item_id", "confidence"], errors="ignore")

# Group by relevant columns and aggregate ingredient names
df_grouped = df.groupby(
    ["restaurant_name", "menu_category", "menu_item", "menu_description",
     "categories", "address1", "city", "zip_code", "country", "state",
     "rating", "review_count", "price"]
)["ingredient_name"].apply(lambda x: ', '.join(x.dropna().unique())).reset_index()

# Save the cleaned data to a new CSV file
output_file_path = "cleaned_menu_data.csv"  # Update as needed
df_grouped.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to {output_file_path}")