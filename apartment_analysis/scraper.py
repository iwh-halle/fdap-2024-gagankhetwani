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
    url = base_url.format(page=page)
    
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
        
        # Extract the price (ensure the correct class and structure)
        price = listing.find('div', class_='col-xs-3 text-right')
        if price:
            price_b = price.find('b')
            listing_data['price'] = get_text_or_default(price_b)
        else:
            listing_data['price'] = 'N/A'
    
        # Extract the availability
        availability = listing.find('div', class_='col-xs-5 text-center')
        listing_data['availability'] = get_text_or_default(availability)
        
        # Extract the size
        size = listing.find('div', class_='col-xs-3 text-center')
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
        
        # Extract the description
        description = listing.find('div', class_='col-xs-11')
        listing_data['description'] = get_text_or_default(description)
        
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
