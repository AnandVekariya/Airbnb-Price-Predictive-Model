# ====================================================
# 🧩 Library Imports
# ====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from word2number import w2n
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

# ====================================================
# 📂 Load Raw Data
# ====================================================
raw_data = pd.read_excel(r"\database\Raw_Final_Project_Data.xlsx")

# ====================================================
# 🌍 Normalize City Names (Belgium)
# ====================================================
city_mapping = {
    "Brussels": [
        "Bruxelles",
        "Ville de Bruxelles",
        "City of Brussels",
        "Brussel",
        "Brüssel",
        "bruxelles",
        "Brussels - Anderlecht",
        "Bruxelles centre ville",
        "Brussels-Centre",
    ],
    "Antwerp": [
        "Antwerpen",
        "Antwerprn",
        "Antwerpen - Ekeren",
        "BORGERHOUT (ANTWERPEN)",
        "Antwerpen Borgerhout",
    ],
    "Ixelles": ["Elsene", "Ixelles/Elsene", "ixelles"],
    "Schaerbeek": ["Schaarbeek", "Bruxelles Schaerbeek", "Schaaerbek"],
    "Saint-Josse-ten-Noode": ["Sint-Joost-ten-Node", "Saint Josse Ten Noode"],
    "Watermael-Boitsfort": ["Watermaal-Bosvoorde", "WATERMAEL BOITSFORT"],
    "Woluwe-Saint-Lambert": ["Woluwé-Saint-Lambert"],
    "Woluwe-Saint-Pierre": [
        "Sint-Pieters-Woluwe",
        "St-Pieters-Woluwe",
        "Woluwe Saint Pierre",
    ],
    "Anderlecht": ["Brussels (Anderlecht)", "Brussels - Anderlecht", "anderlecht"],
    "Laeken": ["Laken", "Laeken"],
}

# Create lowercase lookup
name_to_city = {
    alias.lower(): city for city, aliases in city_mapping.items() for alias in aliases
}


def replace_city_name(city):
    if pd.isna(city):
        return np.nan
    return name_to_city.get(city.lower(), city)


# Apply normalization
raw_data["City"] = raw_data["City"].apply(replace_city_name)


# ====================================================
# 🧹 Data Cleaning & Feature Engineering Functions
# ====================================================
def drop_columns_and_save(df, file_path):
    drop_cols = [
        "Listing Url",
        "Scrape ID",
        "Name",
        "Summary",
        "Space",
        "Description",
        "Neighborhood Overview",
        "Notes",
        "Access",
        "Interaction",
        "House Rules",
        "Thumbnail Url",
        "Medium Url",
        "Picture Url",
        "XL Picture Url",
        "Host URL",
        "Host Name",
        "Host About",
        "Host Thumbnail Url",
        "Host Picture Url",
        "Host Neighbourhood",
        "Host Listings Count",
        "Neighbourhood",
        "Neighbourhood Cleansed",
        "Neighbourhood Group Cleansed",
        "Zipcode",
        "Smart Location",
        "Country Code",
        "Latitude",
        "Longitude",
        "Square Feet",
        "Has Availability",
        "Calendar last Scraped",
        "First Review",
        "Review Scores Accuracy",
        "Review Scores Cleanliness",
        "Review Scores Checkin",
        "Review Scores Communication",
        "Review Scores Location",
        "Review Scores Value",
        "Jurisdiction Names",
        "Geolocation",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    df.to_excel(file_path, index=False)
    return df


def amenities_categories(df):
    categories = {
        "AC/Heating": ["Air Conditioning", "Air conditioning", "Heating"],
        "TV": ["TV", "Cable TV"],
        "Baby Care": [
            "Baby bath",
            "Baby monitor",
            "Babysitter recommendations",
            "Crib",
            "Outlet covers",
            "Pack 'n Play/travel crib",
            "Children's books and toys",
            "Children's dinnerware",
        ],
        "Safety": [
            "Carbon Monoxide Detector",
            "Carbon monoxide detector",
            "Fire Extinguisher",
            "Fire extinguisher",
            "First Aid Kit",
            "First aid kit",
        ],
        "Pets allowed": [
            "Cat(s)",
            "Dog(s)",
            "Other pet(s)",
            "Pets Allowed",
            "Pets allowed",
        ],
        "Kitchen": ["Kitchen", "Microwave", "Cooking basics", "Dishwasher"],
        "Essentials": [
            "Coffee maker",
            "Dishes and silverware",
            "Dryer",
            "Essentials",
            "Firm matress",
            "Firm mattress",
            "Hair Dryer",
            "Hair dryer",
            "Hangers",
            "Hot water",
            "Iron",
            "Keypad",
            "Lock on Bedroom Door",
            "Lock on bedroom door",
            "Bathtub",
        ],
        "Service": [
            "Doorman",
            "Doorman Entry",
            "Extra pillows and blankets",
            "Gym",
            "Host greets you",
            "Laptop Friendly Workspace",
            "Laptop friendly workspace",
            "24-Hour Check-in",
            "24-hour check-in",
            "Bed linens",
            "Breakfast",
            "Buzzer/Wireless Intercom",
            "Buzzer/wireless intercom",
            "Cleaning before checkout",
            "Disabled parking spot",
        ],
        "Facilities": [
            "BBQ grill",
            "EV charger",
            "Elevator in Building",
            "Elevator in building",
            "Ethernet connection",
            "Fireplace guards",
            "Free Parking on Premises",
            "Free parking on premises",
            "Free parking on street",
            "Grab-rails for shower and toilet",
            "High chair",
            "Internet",
            "Lockbox",
            "Long term stays allowed",
            "Luggage dropoff allowed",
            "Paid parking off premises",
            "Accessible-height bed",
            "Accessible-height toilet",
            "Changing table",
            "Flat smooth pathway to front door",
        ],
        "Entertainment": [
            "Game console",
            "BBQ grill",
            "Hot Tub",
            "Hot tub",
            "Indoor Fireplace",
            "Indoor fireplace",
            "Lake access",
            "Path to entrance lit at night",
            "Patio or balcony",
            "Beach essentials",
            "Beachfront",
            "Garden or backyard",
        ],
        "Other Amenities": ["Family/Kid Friendly", "Family/kid friendly", "Other"],
    }

    def has_keyword(amenities, keywords):
        if isinstance(amenities, str):
            return "1" if any(k in amenities for k in keywords) else "0"
        return "0"

    for cat, words in categories.items():
        df[cat] = df["Amenities"].apply(lambda x: has_keyword(x, words))

    return df


def host_location_match(df):
    def match(row):
        if pd.isna(row["Host Location"]):
            return 0
        host_parts = [p.strip().lower() for p in str(row["Host Location"]).split(",")]
        city, street = (
            str(row.get("City", "")).lower(),
            str(row.get("Street", "")).lower(),
        )
        return int(any(p in city or p in street for p in host_parts))

    df["Host Location Same"] = df.apply(match, axis=1)
    return df


def add_availability_flags(df):
    for col in ["Transit", "Market", "Weekly Price", "Monthly Price", "License"]:
        df[f"Has_{col.replace(' ', '_')}"] = df[col].notnull().astype(int)
    return df


def host_age(df):
    for col in ["Last Scraped", "Host Since", "Last Review"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["Host Age"] = (
        ((df["Last Scraped"] - df["Host Since"]).dt.days // 365).fillna(0).astype(int)
    )
    df["Last Rented in month"] = (
        ((df["Last Scraped"] - df["Last Review"]).dt.days // 30).fillna(0).astype(int)
    )
    return df


def categorize_property_type(df):
    mapping = {
        "Apartment": ["Apartment", "Serviced apartment", "Entire Floor"],
        "Townhouse": ["Townhouse"],
        "Condominium": ["Condominium"],
        "House": ["Bed & Breakfast", "House", "Bungalow"],
        "Loft": ["Loft"],
        "Cabin": ["Guesthouse", "Cabin", "In-law"],
        "Dorm": ["Dorm", "Hostel", "Guest suite"],
        "Vacation House/Villa": [
            "Villa",
            "Chalet",
            "Camper/RV",
            "Boat",
            "Castle",
            "Lighthouse",
            "Timeshare",
            "Nature lodge",
            "Earth House",
            "Island",
            "Vacation home",
            "Pension (Korea)",
            "Yurt",
            "Cave",
        ],
        "Tent/Hut": ["Tent", "Igloo", "Hut", "Tipi", "Treehouse"],
        "Other": ["Other", "Boutique hotel", "Parking Space", "Car", "Train"],
    }
    reverse = {ptype: cat for cat, types in mapping.items() for ptype in types}
    df["Property Type"] = df["Property Type"].apply(
        lambda x: reverse.get(x, "Other") if isinstance(x, str) else "Other"
    )
    return df


def process_prices(df):
    def calc_price(row):
        if pd.notna(row["Price"]) and row["Price"] != 0:
            return row["Price"]
        if pd.notna(row["Weekly Price"]):
            return row["Weekly Price"] / 7
        if pd.notna(row["Monthly Price"]):
            return row["Monthly Price"] / 30
        return np.nan

    df["Price"] = df.apply(calc_price, axis=1)

    mean_prices = df.groupby(["Country", "Property Type"])["Price"].mean().reset_index()
    df = df.merge(
        mean_prices, on=["Country", "Property Type"], how="left", suffixes=("", "_mean")
    )

    def gen_price(mean_val):
        if pd.isna(mean_val):
            return 0
        return int(np.random.uniform(mean_val * 0.9, mean_val * 1.1))

    df["Price"] = df["Price"].fillna(df["Price_mean"].apply(gen_price)).astype(int)
    df.drop(columns="Price_mean", inplace=True)
    return df


def update_calendar(df):
    mapping = {
        "today": "In a week",
        "yesterday": "In a week",
        "2 days ago": "In a week",
        "3 days ago": "In a week",
        "4 days ago": "In a week",
        "5 days ago": "In a week",
        "6 days ago": "In a week",
        "1 week ago": "A week ago",
        "a week ago": "A week ago",
        "2 weeks ago": "In a month",
        "3 weeks ago": "In a month",
        "4 weeks ago": "A month ago",
        "5 weeks ago": "A month ago",
        "6 weeks ago": "A month ago",
        "7 weeks ago": "A month ago",
    }
    df["Calendar Updated"] = df["Calendar Updated"].replace(mapping)
    return df


def update_accommodates(df):
    df["Bathrooms"] = df["Bathrooms"].replace("", 1).fillna(1).astype(int)
    df["Bedrooms"] = df["Bedrooms"].replace("", 1).fillna(1).astype(int)
    df["Beds"] = df["Beds"].replace("", 1).fillna(1).astype(int)

    def adjust(row):
        if row["Accommodates"] > (row["Bathrooms"] + row["Bedrooms"]):
            row["Bedrooms"] = int(row["Accommodates"] - row["Bathrooms"])
        return row

    return df.apply(adjust, axis=1)


# ====================================================
# 🏗️ Execute Data Cleaning Pipeline
# ====================================================
df = drop_columns_and_save(
    raw_data,
    r"\database\Raw_Final_Project_Data_1.xlsx",
)

df = (
    df.pipe(amenities_categories)
    .pipe(host_location_match)
    .pipe(add_availability_flags)
    .pipe(host_age)
    .pipe(categorize_property_type)
    .pipe(process_prices)
    .pipe(update_calendar)
    .pipe(update_accommodates)
)

# Drop redundant columns
final_drop_cols = [
    "Last Scraped",
    "Experiences Offered",
    "Transit",
    "Host Since",
    "Host Location",
    "Host Verifications",
    "Amenities",
    "Market",
    "Calendar Updated",
    "Last Review",
    "License",
    "Features",
    "Street",
    "City",
    "State",
]
df = df.drop(columns=final_drop_cols, errors="ignore")

# Save Cleaned Dataset
output_path = r"\database\Cleaning_Final_Clean_Project_Data.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Data cleaning complete. File saved to:\n{output_path}")
