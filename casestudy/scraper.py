import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Function to get the HTML content of a page
def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

# Function to parse a single listing and extract relevant details
def parse_listing(listing):
    details = {}

    try:
        details['price'] = listing.find('div', class_='price').get_text().strip().replace('€', '').replace(',', '.').split()[0]
    except AttributeError:
        details['price'] = None
        
    try:
        details['location'] = listing.find('div', class_='col-xs-11').get_text().strip()
    except AttributeError:
        details['location'] = None
    
    try:
        details['room_size'] = listing.find('div', class_='size').get_text().strip().replace('m²', '').replace(',', '.')
    except AttributeError:
        details['room_size'] = None
    
    try:
        details['num_rooms'] = listing.find('div', class_='rooms').get_text().strip()
    except AttributeError:
        details['num_rooms'] = None
    
    try:
        details['amenities'] = ', '.join([amenity.get_text().strip() for amenity in listing.find_all('span', class_='detail_icon')])
    except AttributeError:
        details['amenities'] = None
    
    try:
        details['listing_date'] = listing.find('div', class_='list_details').find_next('span').get_text().strip()
    except AttributeError:
        details['listing_date'] = None
    
    try:
        details['availability_date'] = listing.find('div', class_='list_details').find_next('span').find_next('span').get_text().strip()
    except AttributeError:
        details['availability_date'] = None
    
    try:
        duration_text = listing.find('div', class_='online_since').get_text().strip().replace('seit', '').replace('Stunden', '').replace('Tagen', '').strip()
        details['duration_online'] = int(duration_text) if 'Tagen' in duration_text else float(duration_text) / 24
    except AttributeError:
        details['duration_online'] = None
    
    return details

# Function to scrape listings from WG-Gesucht
def scrape_wg_gesucht(url, pages=5):
    all_listings = []

    for page in range(1, pages+1):
        page_url = f"{url}?page={page}"
        html_content = get_html(page_url)
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            listings = soup.find_all('div', class_='wgg_card offer_list_item')
            for listing in listings:
                details = parse_listing(listing)
                all_listings.append(details)
        time.sleep(2)  # To avoid overwhelming the server

    return pd.DataFrame(all_listings)

# URL of the WG-Gesucht page for Frankfurt
url = "https://www.wg-gesucht.de/wg-zimmer-in-Frankfurt-am-Main.41.0.1.0.html"

# Scrape the first 5 pages of listings
df = scrape_wg_gesucht(url, pages=5)
df.to_csv('wg_gesucht_listings.csv', index=False)
print(df.head())


