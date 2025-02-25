import pandas as pd

# Load the original CSV file
file_path = "menudata_internal_data.csv"
df = pd.read_csv(file_path)

### **Step 1: Create Restaurants Table**
restaurants = df[['restaurant_name', 'address1', 'city', 'state', 'zip_code', 'country', 'categories', 'rating', 'review_count', 'price']].drop_duplicates()
restaurants = restaurants.reset_index(drop=True)
restaurants.insert(0, 'restaurant_id', range(1, len(restaurants) + 1))

### **Step 2: Create Restaurant Categories Table (Normalized Categories)**
# Extract unique categories and normalize into a separate table
categories_expanded = restaurants[['restaurant_id', 'categories']].drop_duplicates()
categories_expanded = categories_expanded.assign(category=categories_expanded['categories'].str.split('|'))
categories_expanded = categories_expanded.explode('category').drop(columns=['categories'])

### **Step 3: Create Menus Table**
menus = df[['restaurant_name', 'menu_category']].drop_duplicates()
menus = menus.merge(restaurants[['restaurant_id', 'restaurant_name']], on='restaurant_name', how='left')
menus = menus.drop(columns=['restaurant_name']).reset_index(drop=True)
menus.insert(0, 'menu_id', range(1, len(menus) + 1))

### **Step 4: Create Menu Items Table**
menu_items = df[['restaurant_name', 'menu_category', 'item_id', 'menu_item', 'menu_description']].drop_duplicates()
menu_items = menu_items.merge(menus[['menu_id', 'menu_category']], on=['menu_category'], how='left')
menu_items = menu_items.drop(columns=['restaurant_name', 'menu_category']).reset_index(drop=True)

### **Step 5: Create Ingredients Table**
ingredients = df[['item_id', 'ingredient_name', 'confidence']].drop_duplicates()
ingredients = ingredients.reset_index(drop=True)

### **🔹 Save CSV Files for Database Ingestion**
restaurants.drop(columns=['categories']).to_csv("restaurants.csv", index=False)  # Drop categories (now normalized)
categories_expanded.to_csv("restaurant_categories.csv", index=False)
menus.to_csv("menus.csv", index=False)
menu_items.to_csv("menu_items.csv", index=False)
ingredients.to_csv("ingredients.csv", index=False)

print("✅ Data transformation complete! Schema saved as CSVs:")
print("- restaurants.csv")
print("- restaurant_categories.csv (normalized)")
print("- menus.csv")
print("- menu_items.csv")
print("- ingredients.csv")
