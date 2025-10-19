import requests
from bs4 import BeautifulSoup
import json
import re
from difflib import SequenceMatcher

def scrape_nola_council_members():
    """
    Scrapes New Orleans City Council member names from Wikipedia
    Returns a dictionary with current and historical members
    """
    
    url = "https://en.wikipedia.org/wiki/New_Orleans_City_Council"
    
    # Send request to Wikipedia
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Initialize our glossary
    glossary = {
        "current_members": [],
        "historical_members": set(),
        "all_unique_names": set()
    }
    
    # Strategy 1: Find the current members table (the one visible in your screenshot)
    # Look for tables with class "wikitable"
    tables = soup.find_all('table', class_='wikitable')
    
    print(f"Found {len(tables)} wikitable tables")
    
    for idx, table in enumerate(tables):
        print(f"\nProcessing table {idx + 1}...")
        
        # Get all links in this table
        links = table.find_all('a')
        
        for link in links:
            # Get the link text and href
            name = link.get_text().strip()
            href = link.get('href', '')
            
            # Filter criteria:
            # 1. Must have at least 2 words (first and last name)
            # 2. No digits in the name
            # 3. Link should point to a person's Wikipedia page (starts with /wiki/)
            # 4. Exclude common non-name links
            
            exclude_terms = [
                'democratic', 'republican', 'edit', 'citation needed',
                'district', 'council', 'mayor', 'election', 'term',
                'louisiana', 'new orleans', 'vacant', 'appointed'
            ]
            
            if (name and 
                len(name.split()) >= 2 and 
                not any(char.isdigit() for char in name) and
                href.startswith('/wiki/') and
                not any(term in name.lower() for term in exclude_terms) and
                len(name) < 50):  # Reasonable name length
                
                glossary["all_unique_names"].add(name)
                print(f"  Found name: {name}")
    
    # Strategy 2: Look specifically for the "Current members" section
    # Find heading that contains "current members"
    for heading in soup.find_all(['h2', 'h3', 'h4']):
        heading_text = heading.get_text().lower()
        if 'current' in heading_text and 'member' in heading_text:
            print(f"\nFound 'Current members' section: {heading.get_text()}")
            
            # Get the next table after this heading
            next_element = heading.find_next_sibling()
            while next_element:
                if next_element.name == 'table':
                    # Extract names from this specific table
                    rows = next_element.find_all('tr')
                    for row in rows:
                        # Look for links in each row
                        links = row.find_all('a')
                        for link in links:
                            name = link.get_text().strip()
                            if (name and 
                                len(name.split()) >= 2 and 
                                not any(char.isdigit() for char in name)):
                                glossary["current_members"].append(name)
                                glossary["all_unique_names"].add(name)
                    break
                next_element = next_element.find_next_sibling()
    
    # Convert set to sorted list
    glossary["all_unique_names"] = sorted(list(glossary["all_unique_names"]))
    glossary["historical_members"] = sorted(list(glossary["historical_members"]))
    
    # Remove duplicates from current_members
    glossary["current_members"] = list(dict.fromkeys(glossary["current_members"]))
    
    return glossary

# Run the scraper
if __name__ == "__main__":
    print("Starting scraper...\n")
    council_glossary = scrape_nola_council_members()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Total unique names found: {len(council_glossary['all_unique_names'])}")
    print(f"Current members found: {len(council_glossary['current_members'])}")
    
    print(f"\n{'='*60}")
    print("ALL UNIQUE NAMES:")
    print(f"{'='*60}")
    for name in council_glossary['all_unique_names']:
        print(f"  • {name}")
    
    if council_glossary['current_members']:
        print(f"\n{'='*60}")
        print("CURRENT MEMBERS:")
        print(f"{'='*60}")
        for name in council_glossary['current_members']:
            print(f"  • {name}")
    
    # Save to JSON file
    # Convert any remaining sets to lists for JSON serialization
    glossary_for_json = {
        "current_members": council_glossary["current_members"],
        "historical_members": council_glossary["historical_members"],
        "all_unique_names": council_glossary["all_unique_names"]
    }
    
    with open('nola_council_glossary.json', 'w', encoding='utf-8') as f:
        json.dump(glossary_for_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("✓ Glossary saved to 'nola_council_glossary.json'")
    print(f"{'='*60}")