import aiohttp
import asyncio
from bs4 import BeautifulSoup
import pandas as pd
import random

# Base URL for the website with placeholder for page number
base_url = "https://www.wg-gesucht.de/wg-zimmer-in-Frankfurt-am-Main.41.0.1.0.html"

# Initialize a list to store the data
data = []

# Function to clean and extract text
def get_text_or_default(element, default='N/A'):
    return element.get_text(strip=True) if element else default

# Function to extract description from the listing page
async def extract_description(session, listing_url, index):
    async with session.get(listing_url) as response:
        response.raise_for_status()
        response_text = await response.text()
        soup = BeautifulSoup(response_text, 'html.parser')

        description_parts = []

        for section in soup.find_all('div', class_='section_panel_tab'):
            section_id = section['data-text']
            description_div = soup.find('div', id=section_id[1:])
            if description_div:
                description_parts.append(get_text_or_default(description_div.find('p')))
        
        description = ' '.join(description_parts)
        return index, description

# Function to scrape a single page
async def scrape_page(session, page):
    url = base_url.format(page=page)
    async with session.get(url) as response:
        response.raise_for_status()
        response_text = await response.text()
        soup = BeautifulSoup(response_text, 'html.parser')
        
        listings = soup.find_all('div', class_='wgg_card offer_list_item')
        if not listings:
            print(f"No listings found on page {page}, stopping.")
            return False
        
        tasks = []
        for index, listing in enumerate(listings):
            # Initialize a dictionary to store all available data for the listing
            listing_data = {}

            # Extract the full link
            link = listing.find('a', class_='detailansicht')
            full_link = link['href'] if link else 'N/A'
            full_link = "https://www.wg-gesucht.de" + full_link if full_link != 'N/A' else 'N/A'
            listing_data['full_link'] = full_link
            
            # Extract the title
            listing_data['title'] = get_text_or_default(listing.find('h3', class_='truncate_title noprint'))
            
            # Extract the details
            details = listing.find('div', class_='col-xs-11')
            listing_data['details'] = get_text_or_default(details)
            
            # Extract the price
            price_div = listing.find('div', class_='col-xs-3')
            price = price_div.find('b') if price_div else None
            listing_data['price'] = get_text_or_default(price)
            
            # Extract the availability
            availability = listing.find('div', class_='col-xs-5 text-center')
            listing_data['availability'] = get_text_or_default(availability)
            
            # Extract the size
            size_div = listing.find('div', class_='col-xs-3 text-right')
            size = size_div.find('b') if size_div else None
            listing_data['size'] = get_text_or_default(size)
            
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
            
            # Add listing data to the main data list
            data.append(listing_data)
            
            # Extract the description from the full link page
            if full_link != 'N/A':
                tasks.append(extract_description(session, full_link, len(data) - 1))

        # Wait for all description extraction tasks to complete
        descriptions = await asyncio.gather(*tasks)
        for index, description in descriptions:
            data[index]['description'] = description
        
        print(f"Processed page {page}, found {len(listings)} listings.")
        return True

# Main function to run the scraper
async def main():
    async with aiohttp.ClientSession() as session:
        page = 0
        while True:
            success = await scrape_page(session, page)
            if not success:
                break
            delay = random.uniform(2, 5)  # Adjust delay as necessary
            await asyncio.sleep(delay)
            page += 1

# Run the main function
asyncio.run(main())

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file with correct encoding
df.to_csv('wg_gesucht_frankfurt_listings.csv', index=False, encoding='utf-8')

# Print the DataFrame
print(df)