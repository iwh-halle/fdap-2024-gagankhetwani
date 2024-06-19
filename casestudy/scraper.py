import requests
from bs4 import BeautifulSoup

# Send a request to the URL
url = "https://www.wg-gesucht.de/wg-zimmer-in-Frankfurt-am-Main.41.0.1.0.html"
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find all the listing containers
listing_containers = soup.find_all("div", class_="wg_card")

# Initialize an empty list to store the listings
listings = []

# Loop through each listing container and extract the data
for container in listing_containers:
    # Extract the rent
    rent = container.find("span", class_="price").text.strip()
    
    # Extract the size
    size = container.find("span", class_="size").text.strip()
    
    # Extract the number of rooms
    rooms = container.find("span", class_="rooms").text.strip()
    
    # Extract the location
    location = container.find("span", class_="location").text.strip()
    
    # Extract the availability date
    availability = container.find("span", class_="availability").text.strip()
    
    # Extract the listing URL
    listing_url = container.find("a")["href"]
    
    # Append the extracted data to the listings list
    listings.append({
        "rent": rent,
        "size": size,
        "rooms": rooms,
        "location": location,
        "availability": availability,
        "listing_url": listing_url
    })

# Print the extracted listings
for listing in listings:
    print(listing)




