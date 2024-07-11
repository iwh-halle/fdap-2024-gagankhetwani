import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Base URL for the website
base_url = "https://www.wg-gesucht.de/wg-zimmer-in-Frankfurt-am-Main.41.0.1.0.html"

# Number of pages to scrape
num_pages = 5  # Set the number of pages you want to scrape

# Initialize a list to store the data
data = []

# Function to clean and extract text
def get_text_or_default(element, default='N/A'):
    return element.get_text(strip=True) if element else default

# Loop through the specified number of pages
for page in range(1, num_pages + 1):
    # Construct the URL for the current page
    url = base_url + f"?page={page}"
    
    # Make a request to the website
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all listing elements
    listings = soup.find_all('div', class_='wgg_card offer_list_item')
    if not listings:
        print(f"No listings found on page {page}, stopping.")
        break  # Stop if no more listings
    
    # Loop through each listing and extract all information
    for listing in listings:
        # Initialize a dictionary to store all available data for the listing
        listing_data = {}

        # Extract the full link
        link = listing.find('a', class_='detailansicht')
        full_link = link['href'] if link else 'N/A'
        listing_data['full_link'] = "https://www.wg-gesucht.de" + full_link if full_link != 'N/A' else 'N/A'
        
        # Extract the title
        listing_data['title'] = get_text_or_default(listing.find('h3', class_='truncate_title noprint'))
        
        # Extract the details
        details = listing.find('div', class_='col-xs-11')
        listing_data['details'] = get_text_or_default(details)
        
        # Extract the price
        price = listing.find('div', class_='col-xs-3')
        if price:
            price_b = price.find('b')
            listing_data['price'] = get_text_or_default(price_b)
        else:
            listing_data['price'] = 'N/A'
    
        # Extract the availability
        availability = listing.find('div', class_='col-xs-5 text-center')
        listing_data['availability'] = get_text_or_default(availability)
        
        # Extract the size
        size = listing.find('div', class_='col-xs-3 text-right')
        if size:
            size_b = size.find('b')
            listing_data['size'] = get_text_or_default(size_b)
        else:
            listing_data['size'] = 'N/A'
        
        # Extract the landlord name
        landlord = listing.find('span', class_='ml5')
        listing_data['landlord'] = get_text_or_default(landlord)
        
        # Extract the online status
        online_status = 'N/A'
        for span in listing.find_all('span'):
            if 'color' in span.attrs.get('style', ''):
                online_status = span.get_text(strip=True)
                break
        listing_data['online_status'] = online_status
        
        # Append the listing data to the list
        data.append(listing_data)
    
    print(f"Processed page {page}, found {len(listings)} listings.")
    
    # Pause to respect website's request rate
    time.sleep(3)  # Adjust delay as necessary

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file without encoding changes
df.to_csv('wg_gesucht_frankfurt_dynamic_manual_pages.csv', index=False, encoding='utf-8')

# Print the DataFrame
print(df)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, CoxPHFitter

# Load the data
df = pd.read_csv('wg_gesucht_frankfurt_dynamic_manual_pages.csv')

# Print initial data information
print("Initial Data Info:")
print(df.info())
print(df.head())

# Handle missing values (e.g., fill with median for simplicity)
df.fillna(df.median(), inplace=True)

# Convert price and size to numeric values (assuming they are in the format "XXX €" and "XX m²")
df['price'] = df['price'].str.extract('(\d+)').astype(float)
df['size'] = df['size'].str.extract('(\d+)').astype(float)

# Convert availability date to datetime
df['availability'] = pd.to_datetime(df['availability'], errors='coerce')

# Extract features from 'details' if needed (example: extracting number of rooms)
df['num_rooms'] = df['details'].str.extract('(\d+) Zimmer').astype(float)

# Feature engineering: calculate duration online in days (assuming 'online_status' contains this info)
df['duration_online'] = df['online_status'].str.extract('(\d+)').astype(float)

# Display the cleaned dataframe info
print("Cleaned Data Info:")
print(df.info())
print(df.head())

# Summary statistics
print(df.describe())

# Distribution of prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price (€)')
plt.ylabel('Frequency')
plt.show()

# Distribution of room sizes
plt.figure(figsize=(10, 6))
sns.histplot(df['size'], bins=30, kde=True)
plt.title('Distribution of Room Sizes')
plt.xlabel('Size (m²)')
plt.ylabel('Frequency')
plt.show()

# Select only numeric columns for the correlation matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Ensure there are numeric columns to correlate
if not numeric_cols.empty:
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
else:
    print("No numeric columns available for correlation.")

# Define the independent variables (features) and the dependent variable (target)
X = df[['size', 'num_rooms', 'duration_online']].dropna()
y = df.loc[X.index, 'price']

print("Features and target variable data info:")
print(X.info())
print(y.info())

if not X.empty and not y.empty:
    # Add a constant to the independent variables
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Print the summary of the regression model
    print(model.summary())

    # Kaplan-Meier Estimator
    kmf = KaplanMeierFitter()
    kmf.fit(df['duration_online'].dropna(), event_observed=(df['online_status'] != 'N/A').dropna())
    kmf.plot_survival_function()
    plt.title('Survival Function of Listings')
    plt.xlabel('Days Online')
    plt.ylabel('Survival Probability')
    plt.show()

    # Cox Proportional Hazards Model
    cph = CoxPHFitter()
    cph_data = df[['duration_online', 'size', 'num_rooms', 'price']].dropna()
    if not cph_data.empty:
        cph.fit(cph_data, duration_col='duration_online', event_col=(df['online_status'] != 'N/A').loc[cph_data.index])
        cph.plot()
        plt.title('Cox Proportional Hazards Model')
        plt.show()

    # Visualization of regression results
    sns.pairplot(df[['price', 'size', 'num_rooms', 'duration_online']].dropna())
    plt.show()
else:
    print("Insufficient data for regression analysis.")

# Save the cleaned DataFrame
df.to_csv('wg_gesucht_frankfurt_cleaned.csv', index=False)