import pandas as pd
import json
from datetime import datetime

def extract_street_names_from_csv(csv_path='put_path_to_your_csv_here.csv'):
    """
    Extract street names from the New Orleans Master Street Name CSV
    Uses the FULLNAME column which has the complete street name
    """
    print(f"📂 Reading CSV file: {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    print(f"   Found {len(df)} rows in CSV")
    print(f"   Columns: {list(df.columns)}")
    
    # Use the FULLNAME column - it has the complete street name
    # Example: "Avenue A", "Rue Andrée", "Old Chef Menteur Road"
    street_names = df['FULLNAME'].dropna().unique().tolist()
    
    # Clean the names
    cleaned_streets = []
    for street in street_names:
        street_str = str(street).strip()
        if street_str and street_str.lower() not in ['nan', 'none', '']:
            cleaned_streets.append(street_str)
    
    # Remove duplicates and sort
    unique_streets = sorted(list(set(cleaned_streets)))
    
    print(f"   ✓ Extracted {len(unique_streets)} unique street names")
    
    return unique_streets

def merge_council_and_streets(
    council_path='nola_council_glossary.json',
    csv_path='Master_Street_Name.csv',
    output_path='combined_glossary.json'
):
    """
    Merge council member names and street names into one comprehensive glossary
    """
    print("\n" + "="*80)
    print("MERGING COUNCIL MEMBERS + STREET NAMES")
    print("="*80)
    
    # Step 1: Load council member glossary
    print("\n📚 STEP 1: Loading council member glossary...")
    try:
        with open(council_path, 'r', encoding='utf-8') as f:
            council_data = json.load(f)
        
        council_names = council_data.get('all_unique_names', [])
        print(f"   ✓ Loaded {len(council_names)} council member names")
        
        # Print sample
        print(f"   Sample council members:")
        for name in council_names[:5]:
            print(f"      • {name}")
    
    except FileNotFoundError:
        print(f"   ⚠️ {council_path} not found!")
        council_data = {"current_members": [], "historical_members": [], "all_unique_names": []}
        council_names = []
    
    # Step 2: Extract street names from CSV
    print("\n🗺️ STEP 2: Extracting street names from CSV...")
    street_names = extract_street_names_from_csv(csv_path)
    
    # Print sample
    print(f"   Sample street names:")
    for name in street_names[:5]:
        print(f"      • {name}")
    
    # Step 3: Combine everything
    print("\n🔗 STEP 3: Combining all names...")
    
    # Combine both lists
    all_names = council_names + street_names
    
    # Remove duplicates and sort
    unique_all_names = sorted(list(set(all_names)))
    
    print(f"   ✓ Total unique terms: {len(unique_all_names)}")
    
    # Step 4: Create combined glossary with same structure as your original
    combined_glossary = {
        "current_members": council_data.get("current_members", []),
        "historical_members": council_data.get("historical_members", []),
        "all_unique_names": unique_all_names,
        
        # Add metadata for reference (optional)
        "_metadata": {
            "council_member_count": len(council_names),
            "street_name_count": len(street_names),
            "total_unique_count": len(unique_all_names),
            "sources": {
                "council_members": council_path,
                "streets": csv_path
            },
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Step 5: Save combined glossary
    print(f"\n💾 STEP 4: Saving combined glossary to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_glossary, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Saved successfully!")
    
    # Print final statistics
    print("\n" + "="*80)
    print("📊 FINAL STATISTICS")
    print("="*80)
    print(f"Council Members:  {len(council_names)}")
    print(f"Street Names:     {len(street_names)}")
    print(f"Total Unique:     {len(unique_all_names)}")
    print("="*80)
    
    # Print sample of combined names
    print(f"\n📝 Sample of combined glossary (first 20 terms):")
    for i, name in enumerate(unique_all_names[:20], 1):
        print(f"   {i:2d}. {name}")
    
    print("\n✅ MERGE COMPLETE!")
    print(f"Your bot can now use: {output_path}")
    print("="*80 + "\n")
    
    return combined_glossary

if __name__ == "__main__":
    # Run the merge
    # Adjust these paths if your files are in different locations
    merge_council_and_streets(
        council_path='nola_council_glossary.json',
        csv_path='Master_Street_Name.csv',
        output_path='combined_glossary.json'
    )