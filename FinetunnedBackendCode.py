# kayseria_backend.py (or backendCostCalculation.py)
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import google.generativeai as genai
import google.ai.generativelanguage as glm # Keep glm import for Content and Blob types
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from google.generativeai import GenerativeModel
import pandas as pd
from flask_cors import CORS
import base64
import json
import pymongo
from pymongo import MongoClient
from bson import ObjectId
import uuid
import datetime
import requests
import time
import re # Import regex for parsing user queries
import sys # For exiting if API key is missing
import traceback # For detailed error logging
from collections import deque
from CostCalculation import TOKEN_COST_MODEL, TOKEN_COST_VISION


# --- Load Environment Variables ---
load_dotenv()

# --- Global Variables ---
config = {}
db = None
gemini_model = None
vision_model = None
product_catalog = {} # Stores your processed product data
EXCHANGE_RATE = None # Initialize EXCHANGE_RATE globally
PRODUCT_CATALOG_PROMPT_STRING = "" # Global variable for the product catalog prompt
SYSTEM_INSTRUCTION_PROMPT = ""

# --- Session Management Configuration ---
SESSION_TIMEOUT_HOURS = 24  # Session timeout in hours
MAX_CHATS_PER_USER = 50    # Maximum number of chats per user
MAX_MESSAGES_PER_CHAT = 1000  # Maximum messages per chat session

def preprocess_fashion_data(csv_path):
    # The 'global' declarations are fine here, they ensure you modify the global variables.
    global product_catalog, EXCHANGE_RATE, PRODUCT_CATALOG_PROMPT_STRING 

    print(f"Loading product data from {csv_path} in preprocess_fashion_data...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV. DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.") # Debug: check df loaded and its size

        # Print column names to ensure 'SKU' and other expected columns are present
        print(f"CSV Columns: {df.columns.tolist()}")

        catalog_entries = []
        rows_processed_count = 0 # Counter for successfully processed rows
        for index, row in df.iterrows():
            product_id = row['SKU']
            if pd.isna(product_id):
                print(f"Skipping row {index} due to missing SKU: {row.to_dict()}") # Log row if SKU is missing
                continue
            product_id = str(product_id).strip()
            
            # Debug: show product ID being processed
            # print(f"Processing product ID: {product_id} from row {index}") 
            
            # --- START OF PRICE LOGIC (includes previous strikethrough fix) ---
            original_price_pkr = float(row['Original_Price']) if pd.notna(row['Original_Price']) else 0.0
            
            # This variable will always hold the ORIGINAL price
            catalog_original_price_pkr = original_price_pkr 
            
            sale_price_pkr = None # Initialize sale_price_pkr to None
            on_sale = False

            if pd.notna(row['Discount_Price']):
                discount_price_pkr = float(row['Discount_Price'])
                # Only activate 'on_sale' and set 'sale_price_pkr' if there's a genuine discount
                if discount_price_pkr < original_price_pkr: 
                    sale_price_pkr = discount_price_pkr
                    on_sale = True
            # --- END OF PRICE LOGIC ---

            # Parse list-like strings for image_urls, sizes, etc.
            image_urls = [url.strip() for url in row['Images'].split(',')] if pd.notna(row['Images']) else []
            
            # Robust parsing for Item_Size
            sizes_available = []
            if pd.notna(row['Item_Size']):
                try:
                    # Attempt to load as JSON, replacing single quotes with double quotes
                    parsed_size = json.loads(str(row['Item_Size']).replace("'", '"'))
                    if isinstance(parsed_size, list):
                        sizes_available = [str(s).strip() for s in parsed_size]
                    else: # If it's a single value (e.g., "7") that was JSON-parsed
                        sizes_available = [str(parsed_size).strip()]
                except json.JSONDecodeError:
                    # Fallback to splitting by comma if not valid JSON
                    sizes_available = [s.strip() for s in str(row['Item_Size']).split(',')]
            
            # General parsing for comma-separated fields, ensuring they are lists of strings
            colors_available = [c.strip() for c in str(row['Item_Color']).split(',')] if pd.notna(row['Item_Color']) else []
            materials = [m.strip() for m in str(row['Item_Material']).split(',')] if pd.notna(row['Item_Material']) else []
            components = [c.strip() for c in str(row['Item_Type']).split(',')] if pd.notna(row['Item_Type']) else []

            # Ensure all relevant fields are captured for the product catalog
            product_details = {
                'id': product_id,
                'name': row['Name'] if pd.notna(row['Name']) else 'Unknown Product',
                'description': row['Description'] if pd.notna(row['Description']) else 'No description available.',
                'details': row['Description'] if pd.notna(row['Description']) else 'No details available.', # Often same as description
                'category': row['Category'] if pd.notna(row['Category']) else 'Uncategorized',
                'gender': row['Gender'] if pd.notna(row['Gender']) else 'Unisex',
                'image_urls': image_urls,
                'in_stock': True, # Assuming all products in CSV are in stock
                'matches': [], # Placeholder for matching logic
                'on_sale': on_sale,
                'price_pkr': catalog_original_price_pkr, # ORIGINAL price (for strikethrough on frontend)
                'price_usd': round(catalog_original_price_pkr / EXCHANGE_RATE, 2) if EXCHANGE_RATE and catalog_original_price_pkr is not None else None,
                'sale_price_pkr': sale_price_pkr, # Discounted price or None
                'sale_price_usd': round(sale_price_pkr / EXCHANGE_RATE, 2) if EXCHANGE_RATE and sale_price_pkr is not None else None,
                'colors_available': colors_available,
                'sizes_available': sizes_available,
                'materials': materials,
                'components': components,
            }
            product_catalog[product_id] = product_details
            rows_processed_count += 1

            # Build a string for the prompt
            catalog_entries.append(
                f"ID: {product_id}, Name: {product_details['name']}, Category: {product_details['category']}, "
                f"Description: {product_details['description']}, Gender: {product_details['gender']}, "
                f"Colors: {', '.join(product_details['colors_available'])}, Materials: {', '.join(product_details['materials'])}, "
                f"Components: {', '.join(product_details['components'])}, "
                f"Original Price (PKR): {product_details['price_pkr']}" +
                (f", Sale Price (PKR): {product_details['sale_price_pkr']}" if product_details['on_sale'] else "")
            )

        PRODUCT_CATALOG_PROMPT_STRING = "\n".join(catalog_entries)
        print(f"Finished processing CSV. Successfully added {rows_processed_count} products to catalog.") # Debug: final count
        print(f"Total products currently in global catalog: {len(product_catalog)}") # Confirm actual catalog size
        print(f"Length of PRODUCT_CATALOG_PROMPT_STRING: {len(PRODUCT_CATALOG_PROMPT_STRING)} characters.") # Debug: prompt length

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Please ensure it's in the same directory as the script.")
        sys.exit(1) # Exit if CSV is not found
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_path} is empty.")
        sys.exit(1)
    except KeyError as ke:
        print(f"Error: Missing expected column in CSV: {ke}. Please check your CSV headers.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during CSV processing in preprocess_fashion_data: {e}")
        traceback.print_exc() # Print full stack trace for unexpected errors
        sys.exit(1) # Exit if critical data loading fails

def load_config():
    global config, EXCHANGE_RATE, gemini_model, vision_model, db, PRODUCT_CATALOG_PROMPT_STRING, SYSTEM_INSTRUCTION_PROMPT

    # Try loading from config.json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("✅ Config loaded from config.json.")
    except FileNotFoundError:
        print("⚠️ Warning: config.json not found. Attempting to load from environment variables.")
        config = {}

    google_api_key_env = os.getenv('GOOGLE_API_KEY')
    if google_api_key_env:
        config['GOOGLE_API_KEY'] = google_api_key_env
        print("✅ GOOGLE_API_KEY loaded from environment variable.")

    exchange_rate_api_key_env = os.getenv('EXCHANGE_RATE_API_KEY')
    if exchange_rate_api_key_env:
        config['EXCHANGE_RATE_API_KEY'] = exchange_rate_api_key_env
        print("✅ EXCHANGE_RATE_API_KEY loaded from environment variable.")

    # Validate keys
    if 'GOOGLE_API_KEY' not in config:
        print("❌ Error loading config.json: GOOGLE_API_KEY not found in config.json or environment variables.")
        raise ValueError("GOOGLE_API_KEY not found in config.json or environment variables.")

    if 'EXCHANGE_RATE_API_KEY' not in config:
        print("❌ Error loading config.json: EXCHANGE_RATE_API_KEY not found in config.json or environment variables.")
        print("Using a default exchange rate of 280.0 PKR to 1 USD. Please set EXCHANGE_RATE_API_KEY for real-time rates.")
        EXCHANGE_RATE = 280.0
    else:
        try:
            url = f"https://v6.exchangerate-api.com/v6/{config['EXCHANGE_RATE_API_KEY']}/latest/USD"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data['result'] == 'success' and 'PKR' in data['conversion_rates']:
                EXCHANGE_RATE = data['conversion_rates']['PKR']
                print(f"✅ Fetched live exchange rate: 1 USD = {EXCHANGE_RATE} PKR.")
            else:
                print("❌ Failed to fetch live exchange rate (PKR not found or result not success). Using default 280.0.")
                EXCHANGE_RATE = 280.0
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching exchange rate: {e}. Using default 280.0.")
            EXCHANGE_RATE = 280.0
        except json.JSONDecodeError:
            print("❌ Error decoding exchange rate response. Using default 280.0.")
            EXCHANGE_RATE = 280.0

    # --- Initialize MongoDB ---
    try:
        mongo_uri = os.getenv('MONGODB_URI')
        database_name = os.getenv('MONGODB_DATABASE', 'fashion_bot')
        
        # If not in environment, try to read from config.json
        if not mongo_uri:
            try:
                with open('config.json', 'r') as f:
                    config = json.load(f)
                mongo_uri = config.get('MONGODB_URI', 'mongodb://localhost:27017/')
                database_name = config.get('MONGODB_DATABASE', 'fashion_bot')
            except Exception as e:
                print(f"⚠️  Could not read config.json: {e}")
                mongo_uri = 'mongodb://localhost:27017/'
                database_name = 'fashion_bot'
        
        if not mongo_uri:
            raise ValueError("MONGODB_URI not found in environment variables or config.json.")
        
        client = MongoClient(mongo_uri)
        db = client[database_name]
        print("✅ MongoDB initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing MongoDB: {e}")
        db = None

    # --- LOCATION FOR CALLING preprocess_fashion_data ---
    csv_file_path = os.getenv('CSV_FILE_PATH', 'Kayseria_clothing_updated.csv')
    preprocess_fashion_data(csv_file_path)

    # --- Construct the GLOBAL SYSTEM_INSTRUCTION_PROMPT after catalog is loaded ---
    # This ensures the product catalog is always part of the model's core instructions.
    SYSTEM_INSTRUCTION_PROMPT = f"""
You are a helpful and friendly AI Fashion Bot for a wide range of clothing and accessories for both men and women. Your offerings include women's dresses, shirts, trousers, and suits, men's shirts and trousers, and unisex shoes, jewelry, watches, perfumes, and bags.
Your goal is to provide personalized style advice, product recommendations, and answer questions about fashion items, considering gender and product type.

Here is the product catalog. Use this exact information to provide recommendations and answer questions.
If a specific detail is not in the catalog for a product, state it as "N/A" or "Not specified".
Do NOT invent products, prices, colors, materials, or any other details not explicitly present in the *conceptual* product catalog.

Product Catalog:
 {PRODUCT_CATALOG_PROMPT_STRING}

Capabilities:
- **Crucial Category Mapping:** If the user asks for general terms like "dresses", "clothes", "suits", or "outfits", do NOT say you don't have them. Instead, understand these as referring to categories like "Ready To Wear", "Unstitched", "2 Piece Suits", "3 Piece Suits", "Lawn Collection", "Shirts", "Trousers", etc. Proactively suggest these specific categories to the user and ask them which specific type they are interested in (e.g., "We have a beautiful collection of Ready To Wear, Unstitched, 2 Piece Suits, and 3 Piece Suits. Which one would you like to explore?").
- Recommend outfits for various occasions (e.g., "party wear dresses", "casual outfits for summer", "formal attire for men").
- Suggest matching items across categories and genders (e.g., "men's watches with a suit", "jewelry with a dress", "shoes for an outfit", "bag for an event").
- Provide information on new arrivals.
- **Identify and recommend products that are currently on sale.** Clearly state if a product is on sale and provide both the discounted and original prices.
- **Filter products by price:** Respond to requests like "show dresses under 20,000", "items between 5000 and 10000", "shirts over 8000".
- **Filter products by category/occasion:** Respond to requests like "show lawn dresses", "formal wear for women", "party wear shoes".
- **Combine filters:** Handle requests like "show lawn dresses under 5000", "formal shirts for men on sale".
- Answer questions about discounts, prices, colors, fabrics, and product descriptions for all product types.
- **Enhanced Conversational Tone:** When providing advice or answering questions, be very empathetic, encouraging and detailed. For example, if a user asks if an item can be worn in winter, explain the fabric suitability and assure them they will look beautiful and stunning. Make the user feel confident and well-informed.
- **CRITICAL: When filtering or recommending, strictly adhere to the exact attributes (e.g., color, material, type, gender) specified in the user's request. Prioritize exact matches over general relevance if a specific attribute is mentioned.**
- **CRITICAL: When recommending Male products make sure that you only suggest products with gender male. Do NOT suggest male products with female or female products with male.**
- **CRITICAL: When recommending products in your natural language response, DO NOT include category, gender, price, color, materials, or any other specific details, including the product's Name. These details will be visible in the dedicated product cards displayed on the frontend. Just provide a general conversational reply like 'I've found some lovely options for you!'.**
- **DO NOT include Product IDs or SKUs directly in your natural language response body.** These are for backend processing only.
- When recommending on sale products make sure that you only recommend that had discount price, Do NOT recommend products that do not have discounted prices.
- When recommending outfit/dresses, suggest relevant accessories separately.
- **CRITICAL: When user ask about specific selected products, only show and talks about that selected specific products and donot include anyother products in your response.**
- When user ask about total price of selected products, you should calculate the total price of selected products and show it to the user.


- **Product Recommendation Strategy (CRITICAL):**
    - Your primary goal is to provide product recommendations.
    - **Crucially, when you recommend main products (like outfits or dresses), you MUST also recommend a small number of complementary items (watches, bags, shoes, accessories) that are logically paired with the main products.**
    - You can recommend a maximum of 6 main products.
    - You can recommend up to 6 suggestive/complementary products.
    - **DO NOT include any introductory phrases about "suggestive products" or "complementary items" in your natural language response body.** The backend will handle this text. Just provide the main conversational reply and the product IDs in the structured JSON.
    

- **Image Input Handling (CRITICAL - READ CAREFULLY):**
  - When a user provides an image, your IMMEDIATE and PRIMARY task is to **analyze the image to identify any specific fashion items or products that match entries in the 'Internal Product Catalog Index' provided to you.**
  - **If you confidently identify a product from your catalog within the image, you MUST:**
    1.  Acknowledge that you've found something relevant in your natural language response (e.g., "I see something interesting in your image!").
    2.  Then, ask the user a follow-up question about what they want to do with it (e.g., "Are you looking for something similar?", "Would you like to know more about it?", "How can I assist you with this item?").
    3.  **MOST IMPORTANTLY: Include the exact Product ID of the identified product in the `PRODUCTS_LIST` under the "main" key at the end of your response.**
  - **If you cannot find an exact match for a product in the image from your catalog, but you can clearly describe the item (e.g., "I see an elegant blue dress" or "I see a traditional men's outfit"), then you MUST:**
    1.  Describe what you see in the image (generally, without specific product names if possible).
    2.  Proactively suggest similar items from your catalog based on its category. You can mention categories like "dresses" or "men's outfits" but avoid specific product names.
    3.  Ask the user for clarification or guidance (e.g., "I see an elegant blue dress in your image. While I don't have an exact match, we have many beautiful blue dresses. Would you like to explore those, or can you tell me more about what you're looking for?").
    4.  **If you suggest similar categories or general items from the catalog, include their relevant IDs in the `PRODUCTS_LIST` under the "main" key.**
  - **NEVER state that you cannot access visual information or that your catalog doesn't include image details. Always try to process the image and provide a relevant, helpful response.**

- **CRITICAL BACKEND SIGNAL (IMPORTANT CHANGE IN FORMAT):**
    Whenever you recommend or list specific products (either from text query or identified in an image), you MUST include their IDs in a structured JSON format at the very end of your natural language response, like this:
    `PRODUCTS_LIST: {{"main": ["ID_MAIN_1", "ID_MAIN_2"], "suggestive": ["ID_SUGGESTIVE_1", "ID_SUGGESTIVE_2"]}}`
    - "main" should contain IDs for primary recommendations (e.g., outfits, dresses, shirts).
    - "suggestive" should contain IDs for complementary items (e.g., perfumes, watches, jewelry, bags, shoes, accessories).
    - If there are no products of a certain type, the list should be empty (e.g., `"main": []` or `"suggestive": []`).
    - **This entire `PRODUCTS_LIST` string will be REMOVED by the backend before display to the user.**

Response Format:
Your response should primarily be natural language. If you recommend specific products, include their IDs in the structured JSON format at the end of your natural language response, like this:
"[Your natural language response here. This is what the user will see.]
PRODUCTS_LIST: {{"main": ["ID1"], "suggestive": ["ID2", "ID3"]}}"
 
Example response (if you identify 'Turquoise Khaddar Tunic 2 piece' from image and suggest a watch):
"I see something beautiful in your image! How can I assist you with this item?
PRODUCTS_LIST: {{"main": ["KSY-2PC-004"], "suggestive": ["M-WATCH-001"]}}"

Example response (if image shows a men's shirt, but no exact match is found, suggesting similar, no accessories):
"I see a stylish men's shirt in your image. While I don't have an exact match, we have many great options. Would you like to explore those?
PRODUCTS_LIST: {{"main": ["M-SHIRT-001", "M-SHIRT-002"], "suggestive": []}}"

Example response (when user asks for "dresses" generally, or shows an image of a dress but no specific product is found and suggests categories):
"I see a beautiful dress in your image! We have a beautiful collection of **Ready To Wear**, **Unstitched**, **2 Piece Suits**, and **3 Piece Suits**. Which one would you like to explore or what would you like to know about the dress in the image?
PRODUCTS_LIST: {{"main": ["2PC-001", "2PC-002", "2PC-003"], "suggestive": []}}"

"""
    print("✅ Gemini API configured with system instruction.")


    # Configure Gemini API with the system_instruction
    try:
        genai.configure(api_key=config['GOOGLE_API_KEY'])
        # Pass system_instruction directly to the GenerativeModel constructor
        gemini_model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_INSTRUCTION_PROMPT)
        vision_model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_INSTRUCTION_PROMPT)
        print("✅ Gemini API configured with system instruction.")
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        sys.exit(1)

    # --- DEBUGGING PRINTS (added new prints for SYSTEM_INSTRUCTION_PROMPT) ---
    print(f"\n--- DEBUG: After product catalog loading ---")
    print(f"Number of products loaded into product_catalog: {len(product_catalog)}")
    print("First 3 products in product_catalog:")
    for i, (prod_id, prod_data) in enumerate(product_catalog.items()):
        if i >= 3: break
        print(f"  ID: {prod_id}, Name: {prod_data.get('name')}, Price: {prod_data.get('price_pkr')}")
    print(f"Length of PRODUCT_CATALOG_PROMPT_STRING: {len(PRODUCT_CATALOG_PROMPT_STRING)}")
    print("Snippet of PRODUCT_CATALOG_PROMPT_STRING (first 500 chars):")
    print(PRODUCT_CATALOG_PROMPT_STRING[:500])
    print(f"Length of SYSTEM_INSTRUCTION_PROMPT: {len(SYSTEM_INSTRUCTION_PROMPT)}")
    print("Snippet of SYSTEM_INSTRUCTION_PROMPT (first 500 chars):")
    print(SYSTEM_INSTRUCTION_PROMPT[:500])
    print(f"--- END DEBUG: Product ID Extraction ---\n")



# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes


load_config() # Load config at startup, which also configures Gemini API

# --- Gemini Pro Vision Model Initialization ---
# This block is somewhat redundant if vision_model is already set in load_config
# and is the same model. You can probably remove it if load_config fully handles it.
# If you keep it, ensure it doesn't try to re-initialize a model already initialized.
try:
    if vision_model is None: # Only try to initialize if not already done by load_config
        vision_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini Vision Model initialized (secondary check).")
except Exception as e:
    print(f"❌ Error initializing Gemini Vision Model (secondary check): {e}")
    vision_model = None


def get_current_exchange_rate():
    """Fetches and caches the USD to PKR exchange rate."""
    # This function is now redundant as exchange rate is fetched in load_config
    # and stored in global EXCHANGE_RATE.
    return EXCHANGE_RATE # Return the global EXCHANGE_RATE


# --- Product Data Loading and Preprocessing ---
# REMOVE THIS LINE: csv_file_path = os.getenv('CSV_FILE_PATH', 'Kayseria_clothing.csv')
# REMOVE THIS LINE: product_catalog = {} # This re-initializes it, use the global one

# --- Helper Functions (Updated/New) ---

def get_product_details_for_ai(product_id):
    """Returns a simplified, text-based description of a product for the AI."""
    product = product_catalog.get(product_id)
    if not product:
        return f"Product with ID {product_id} not found."

    details = [f"Product ID: {product['id']}", f"Name: {product['name']}", f"Category: {product['category']}", f"Gender: {product['gender']}"]
    if product['on_sale']:
        details.append(f"Sale Price: PKR {product['sale_price_pkr']:.2f} (Original: PKR {product['price_pkr']:.2f})")
    else:
        details.append(f"Price: PKR {product['price_pkr']:.2f}")

    if product['description']:
        details.append(f"Description: {product['description']}")
    if product['materials']:
        details.append(f"Materials: {product['materials']}")
    if product['components']:
        details.append(f"Components: {product['components']}")
    if product['details']:
        details.append(f"Additional Details: {product['details']}")
    if product['sizes_available']:
        details.append(f"Available Sizes: {', '.join(product['sizes_available'])}")
    if product['colors_available']:
        details.append(f"Available Colors: {', '.join(product['colors_available'])}")
    details.append(f"In Stock: {'Yes' if product['in_stock'] else 'No'}")
    return "\n".join(details)

# Assuming product_catalog and EXCHANGE_RATE are defined globally or accessible in this scope.
# Make sure EXCHANGE_RATE is correctly defined, e.g., EXCHANGE_RATE = 283.45

def get_recommended_products_data(product_ids):
    """
    Returns a list of full product details for product IDs,
    using pre-defined prices (original or discounted) from the catalog.
    """
    products_for_frontend = []
    for pid in product_ids:
        product_raw = product_catalog.get(pid)
        if product_raw:
            # Create a copy to avoid modifying the original catalog entry
            product_data = product_raw.copy()

            # The 'price_pkr' in product_catalog already holds the current effective price (original or discounted).
            # The 'sale_price_pkr' and 'on_sale' also hold the pre-calculated sale info.
            # So, we just need to ensure the USD prices are available.

            # Ensure price_usd is calculated based on the *effective* price (price_pkr)
            # which is already set to discounted price in preprocess_fashion_data if applicable.
            effective_price_pkr = product_data.get('price_pkr') # This holds the current selling price (discounted if on sale)
            if isinstance(effective_price_pkr, (int, float)) and 'EXCHANGE_RATE' in globals() and EXCHANGE_RATE != 0:
                product_data['price_usd'] = round(effective_price_pkr / EXCHANGE_RATE, 2)
            else:
                product_data['price_usd'] = None

            # Calculate sale_price_usd if the product is on sale
            if product_data.get('on_sale') and product_data.get('sale_price_pkr') is not None:
                sale_price_pkr_from_catalog = product_data['sale_price_pkr']
                if isinstance(sale_price_pkr_from_catalog, (int, float)) and 'EXCHANGE_RATE' in globals() and EXCHANGE_RATE != 0:
                    product_data['sale_price_usd'] = round(sale_price_pkr_from_catalog / EXCHANGE_RATE, 2)
                else:
                    product_data['sale_price_usd'] = None # Cannot calculate USD if rate is missing/zero
            else:
                product_data['sale_price_usd'] = None # Not on sale, so no sale price in USD
            products_for_frontend.append(product_data)
            
    return products_for_frontend

def get_token_costs(input_tokens, output_tokens, exchange_rate):
    input_cost_usd = input_tokens * TOKEN_COST_MODEL["gemini-2.5-flash"]["input"] # Use correct constant
    output_cost_usd = output_tokens * TOKEN_COST_MODEL["gemini-2.5-flash"]["output"] # Use correct constant
    total_cost_usd = input_cost_usd + output_cost_usd
    total_cost_pkr = total_cost_usd * exchange_rate
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "total_cost_usd_turn": total_cost_usd,
        "total_cost_pkr_turn": total_cost_pkr
    }

def get_or_create_user_session(user_id):
    """Retrieves user session data from MongoDB, or initializes if new."""
    if db is None:
        print("MongoDB not initialized, cannot get or create user session.")
        return {
            "history": [],
            "total_cost_usd": 0.0,
            "total_cost_pkr": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_image_displays": 0
        }
    
    user_collection = db.user_sessions
    
    # Check for session timeout and cleanup old sessions
    timeout_threshold = datetime.datetime.now() - datetime.timedelta(hours=SESSION_TIMEOUT_HOURS)
    user_collection.delete_many({
        "user_id": user_id,
        "last_updated": {"$lt": timeout_threshold}
    })
    
    user_doc = user_collection.find_one({"user_id": user_id})
    if user_doc:
        # Convert MongoDB ObjectId to string for JSON serialization
        user_doc['_id'] = str(user_doc['_id'])
        
        # Clean up existing history that might contain full product data
        user_doc = cleanup_history_product_data(user_doc)
        
        # Update last_updated timestamp
        user_collection.update_one(
            {"user_id": user_id},
            {"$set": {"last_updated": datetime.datetime.now()}}
        )
        
        print(f"Retrieved session for {user_id}. Total history messages: {len(user_doc.get('history', []))}")
        return user_doc
    else:
        # Initialize new session with enhanced metadata
        initial_data = {
            "user_id": user_id,
            "history": [],
            "total_cost_usd": 0.0,
            "total_cost_pkr": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_image_displays": 0,
            "session_id": str(uuid.uuid4()),  # Unique session identifier
            "created_at": datetime.datetime.now(),
            "last_updated": datetime.datetime.now(),
            "session_metadata": {
                "user_agent": request.headers.get('User-Agent', 'Unknown'),
                "ip_address": request.remote_addr,
                "session_type": "web_chat"
            }
        }
        user_collection.insert_one(initial_data)
        print(f"Created new session for {user_id} with session_id: {initial_data['session_id']}")
        return initial_data

def update_user_session(user_id, session_data):
    """Updates user session data in MongoDB."""
    if db is None:
        print("MongoDB not initialized, cannot update user session.")
        return

    user_collection = db.user_sessions
    # Prepare a safe update payload that won't try to modify immutable fields like _id
    session_data_to_update = dict(session_data) if isinstance(session_data, dict) else {}
    # Remove immutable _id if present (it may have been stringified earlier)
    session_data_to_update.pop('_id', None)
    # Never update the user_id field via $set to avoid conflicts
    session_data_to_update.pop('user_id', None)
    # Treat created_at as immutable; only set it on insert
    session_data_to_update.pop('created_at', None)

    # Ensure timestamps are handled correctly before updating
    # MongoDB can handle datetime objects directly.
    # If they are ISO strings, convert back to datetime objects for MongoDB.
    if 'created_at' in session_data_to_update and isinstance(session_data_to_update['created_at'], str):
        try:
            session_data_to_update['created_at'] = datetime.datetime.fromisoformat(session_data_to_update['created_at'])
        except ValueError:
            pass  # Keep as string if invalid format

    # Always update last_updated to datetime.datetime.now() to ensure freshness
    session_data_to_update['last_updated'] = datetime.datetime.now()

    # Split 'history' out to avoid path conflicts in a single update document
    history_value = session_data_to_update.pop('history', None)

    # First update: everything except 'history'
    if session_data_to_update:
        user_collection.update_one(
            {"user_id": user_id},
            {
                "$set": session_data_to_update,
                "$setOnInsert": {
                    "user_id": user_id,
                    "created_at": datetime.datetime.now(),
                },
            },
            upsert=True,
        )

    # Second update: set 'history' if provided
    if history_value is not None:
        user_collection.update_one(
            {"user_id": user_id},
            {"$set": {"history": history_value, "last_updated": datetime.datetime.now()}},
            upsert=False,
        )
    print(f"Updated session for {user_id}.")

def cleanup_history_product_data(session_data):
    """
    Cleans up existing history that contains full product data and replaces with just IDs.
    This helps reduce token usage for existing sessions.
    """
    if not session_data or 'history' not in session_data:
        return session_data
    
    cleaned_history = []
    for message in session_data['history']:
        cleaned_message = message.copy()
        
        # Remove full product data if present
        if 'main_recommended_products' in cleaned_message:
            del cleaned_message['main_recommended_products']
        if 'suggestive_recommended_products' in cleaned_message:
            del cleaned_message['suggestive_recommended_products']
        
        # Keep only the essential data
        essential_keys = ['role', 'type', 'text', 'timestamp', 'cost_info']
        if message.get('type') == 'image':
            essential_keys.extend(['image'])
        
        cleaned_message = {k: v for k, v in cleaned_message.items() if k in essential_keys}
        cleaned_history.append(cleaned_message)
    
    session_data['history'] = cleaned_history
    return session_data

def _get_gemini_history(session_data):
    """
    Converts stored session history into the format expected by Gemini's GenerativeModel.
    Handles both text and image parts, decoding image data from base64.
    """
    history_contents = []
    for message in session_data['history']:
        parts = []

        # Text from user or model
        if message.get('type') in ['text', 'model_response'] and message.get('text'):
            text = message['text']
            if isinstance(text, str) and text.strip():
                parts.append(glm.Part(text=text.strip()))

        # Image from user
        elif message.get('type') == 'image':
            image_info = message.get('image', {})
            if image_info.get('data') and image_info.get('mime_type'):
                try:
                    image_bytes = base64.b64decode(image_info['data'])
                    parts.append(glm.Part(
                        inline_data=glm.Blob(
                            mime_type=image_info['mime_type'],
                            data=image_bytes
                        )
                    ))
                except Exception as e:
                    print(f"⚠️ Skipping invalid image part: {e}")
        
        # Only append to history if valid parts and role exist
        if parts and message.get('role') in ['user', 'model']:
            history_contents.append(glm.Content(role=message['role'], parts=parts))

    return history_contents


def calculate_image_cost(image_count, exchange_rate):
    """Calculates the cost of processing images."""
    # TOKEN_COST_VISION['image'] is the cost per image
    total_image_cost_usd = image_count * TOKEN_COST_VISION['image']
    total_image_cost_pkr = total_image_cost_usd * exchange_rate
    return {
        "image_cost_usd": total_image_cost_usd,
        "image_cost_pkr": total_image_cost_pkr
    }


def update_total_costs(session_data, turn_costs):
    """Updates the total accumulated costs in the session data."""
    session_data['total_cost_usd'] = session_data.get('total_cost_usd', 0.0) + turn_costs.get('total_cost_usd_turn', 0.0)
    session_data['total_cost_pkr'] = session_data.get('total_cost_pkr', 0.0) + turn_costs.get('total_cost_pkr_turn', 0.0)
    session_data['total_input_tokens'] = session_data.get('total_input_tokens', 0) + turn_costs.get('input_tokens', 0)
    session_data['total_output_tokens'] = session_data.get('total_output_tokens', 0) + turn_costs.get('output_tokens', 0)
    if 'image_cost_usd' in turn_costs:
        session_data['total_cost_usd'] += turn_costs['image_cost_usd']
        session_data['total_cost_pkr'] += turn_costs['image_cost_pkr']
        session_data['total_image_displays'] = session_data.get('total_image_displays', 0) + 1 # Increment image count

    return session_data

def get_gemini_response(convo_history, current_user_message_parts, model_name='gemini-pro'):
    """
    Sends conversation history to Gemini and gets a response.
    
    IMPORTANT: Token calculation includes ALL tokens sent to Gemini (history + current message),
    not just the current user message. This is because Gemini processes the entire conversation
    context, including stored product data in the history.
    """
    # Define a maximum number of messages to keep in history
    MAX_HISTORY_MESSAGES = 1  # Reduced from 2 to minimize token usage
    
    try:
        # Determine model based on content (image = vision model)
        is_vision = any(
            isinstance(part, glm.Part) and hasattr(part, "inline_data")
            for part in current_user_message_parts
        )
        model = vision_model if is_vision else gemini_model
        print(f"Using {model.model_name} for {'multimodal' if is_vision else 'text'} input.")
        
        # 1. First, fetch the raw history from your system.
        #    This is the part that was likely causing the error in your previous code.
        #    We use the convo_history passed to this function.
        raw_convo_history = convo_history
        
        # 2. Correctly format the history messages for the Gemini API
        formatted_history = []
        for message in raw_convo_history:
            if isinstance(message, dict):
                # If the message is a dictionary, assume it has 'role' and 'parts' keys
                formatted_history.append(glm.Content(
                    role=message.get('role'),
                    parts=message.get('parts')
                ))
            elif isinstance(message, list) and len(message) >= 2:
                # If the message is a list, assume index 0 is role and 1 is parts
                formatted_history.append(glm.Content(
                    role=message[0],
                    parts=message[1]
                ))

        # 3. Create a deque to manage a fixed-size history
        #    This limits the history to the most recent messages.
        history_deque = deque(formatted_history, maxlen=MAX_HISTORY_MESSAGES)
        
        # TEMPORARY: Disable conversation history completely to reduce token usage
        print("🚨 TEMPORARILY DISABLED CONVERSATION HISTORY TO REDUCE TOKEN USAGE")
        history_deque.clear()
        
        # Check if history would cause too many tokens and disable if needed
        if list(history_deque):
            estimated_history_tokens = model.count_tokens([part for content in list(history_deque) for part in content.parts]).total_tokens
            current_message_tokens = model.count_tokens(current_user_message_parts).total_tokens
            estimated_total = estimated_history_tokens + current_message_tokens
            
            if estimated_total > 15000:  # More aggressive limit - reduced from 25000
                print(f"⚠️  Estimated tokens too high ({estimated_total}), disabling conversation history")
                history_deque.clear()  # Clear history to avoid token limit issues

        # 4. Wrap the user message in Content object
        user_message_content = glm.Content(role="user", parts=current_user_message_parts)

        # 5. Start a new chat with the limited history
        chat_session = model.start_chat(history=list(history_deque))

        # 6. Send current message
        response = chat_session.send_message(user_message_content)

        # 7. Calculate input tokens for the current turn. This will now be a small, fixed number.
        # FIXED: Count ALL tokens being sent to Gemini, not just current user message
        # This includes conversation history + current user message
        all_input_parts = []
        
        # Add conversation history parts
        for content in list(history_deque):
            if hasattr(content, 'parts'):
                all_input_parts.extend(content.parts)
        
        # Add current user message parts
        all_input_parts.extend(current_user_message_parts)
        
        # Count total input tokens
        input_tokens = model.count_tokens(all_input_parts).total_tokens
        
        # Debug: Show token breakdown
        history_tokens = model.count_tokens([part for content in list(history_deque) for part in content.parts]).total_tokens if list(history_deque) else 0
        current_message_tokens = model.count_tokens(current_user_message_parts).total_tokens
        
        print(f"🔍 Token Breakdown:")
        print(f"   - History tokens: {history_tokens}")
        print(f"   - Current message tokens: {current_message_tokens}")
        print(f"   - Total input tokens: {input_tokens}")
        print(f"   - History messages count: {len(list(history_deque))}")
        
        # Warning for high token usage
        if input_tokens > 30000:
            print(f"⚠️  WARNING: High input token count ({input_tokens}). This may be due to:")
            print(f"   - Large conversation history with product data")
            print(f"   - Multiple product recommendations in history")
            print(f"   - Consider reducing MAX_HISTORY_MESSAGES (currently {MAX_HISTORY_MESSAGES})")
        
        # 8. Calculate output tokens
        output_tokens = model.count_tokens([response.candidates[0].content]).total_tokens
        
        print(f"Input tokens for this turn: {input_tokens}")
        print(f"Output tokens for this turn: {output_tokens}")

        return response.text, input_tokens, output_tokens

    except Exception as e:
        print(f"❌ Error getting Gemini response: {e}")
        return "Sorry, something went wrong while processing your request.", 0, 0


def extract_product_ids_from_response(text):
    """
    Extracts product IDs from the AI's response using the new structured JSON format.
    Returns a dictionary like {"main": ["ID1", "ID2"], "suggestive": ["ID3", "ID4"]}.
    If format not found, returns empty lists.
    """
    products_list_pattern = r'PRODUCTS_LIST:\s*(\{.*\})'
    match = re.search(products_list_pattern, text, re.DOTALL) # re.DOTALL to match across newlines

    main_ids = []
    suggestive_ids = []

    if match:
        json_string = match.group(1)
        try:
            product_dict = json.loads(json_string)
            if isinstance(product_dict, dict):
                main_ids = [str(pid).strip() for pid in product_dict.get('main', []) if str(pid).strip()]
                suggestive_ids = [str(pid).strip() for pid in product_dict.get('suggestive', []) if str(pid).strip()]
        except json.JSONDecodeError as e:
            print(f"Warning: Could not decode PRODUCTS_LIST JSON: {e}")
            print(f"JSON string: {json_string}")
        except Exception as e:
            print(f"Error parsing PRODUCTS_LIST: {e}")
            
    return {"main": main_ids, "suggestive": suggestive_ids}


# --- Multi-Chat Functions ---

def create_new_chat(user_id, chat_name="New Chat"):
    """Creates a new chat session for a user with enhanced session management."""
    if db is None:
        return None
    
    chat_collection = db.user_chats
    
    # Check if user has reached maximum chat limit
    existing_chats_count = chat_collection.count_documents({"user_id": user_id})
    if existing_chats_count >= MAX_CHATS_PER_USER:
        print(f"User {user_id} has reached maximum chat limit ({MAX_CHATS_PER_USER})")
        return None
    
    # Get user session for context
    user_session = get_or_create_user_session(user_id)
    
    new_chat = {
        "user_id": user_id,
        "chat_name": chat_name,
        "chat_session_id": str(uuid.uuid4()),  # Unique chat session identifier
        "user_session_id": user_session.get('session_id'),  # Link to user session
        "created_at": datetime.datetime.now(),
        "last_updated": datetime.datetime.now(),
        "messages": [],
        "chat_metadata": {
            "total_messages": 0,
            "last_message_at": None,
            "chat_type": "fashion_assistant",
            "created_from_session": user_session.get('session_id')
        }
    }
    
    result = chat_collection.insert_one(new_chat)
    new_chat["_id"] = str(result.inserted_id)
    
    print(f"Created new chat session {new_chat['chat_session_id']} for user {user_id}")
    return new_chat

def get_user_chats(user_id):
    """Retrieves all chat sessions for a user."""
    if db is None:
        return []
    
    chat_collection = db.user_chats
    chats = list(chat_collection.find({"user_id": user_id}).sort("last_updated", -1))
    
    # Convert ObjectId to string for JSON serialization
    for chat in chats:
        chat["_id"] = str(chat["_id"])
        if "created_at" in chat and hasattr(chat["created_at"], "isoformat"):
            chat["created_at"] = chat["created_at"].isoformat()
        if "last_updated" in chat and hasattr(chat["last_updated"], "isoformat"):
            chat["last_updated"] = chat["last_updated"].isoformat()
    
    return chats

def get_chat_messages(chat_id, user_id):
    """Retrieves messages for a specific chat with session validation."""
    if db is None:
        return []
    
    chat_collection = db.user_chats
    chat = chat_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
    
    if not chat:
        print(f"Chat {chat_id} not found or doesn't belong to user {user_id}")
        return []
    
    # Validate chat session is still active
    chat_session_id = chat.get("chat_session_id")
    if not chat_session_id:
        print(f"Chat {chat_id} has no valid session ID")
        return []
    
    # Convert ObjectId to string for JSON serialization
    chat["_id"] = str(chat["_id"])
    if "created_at" in chat and hasattr(chat["created_at"], "isoformat"):
        chat["created_at"] = chat["created_at"].isoformat()
    if "last_updated" in chat and hasattr(chat["last_updated"], "isoformat"):
        chat["last_updated"] = chat["last_updated"].isoformat()
    
    messages = chat.get("messages", [])
    print(f"Retrieved {len(messages)} messages from chat {chat_id} (session: {chat_session_id})")
    
    return messages

def add_message_to_chat(chat_id, user_id, message_data):
    """Adds a new message to a specific chat with enhanced session management."""
    if db is None:
        return False
    
    chat_collection = db.user_chats
    
    # Validate chat exists and belongs to user
    chat = chat_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
    if not chat:
        print(f"Chat {chat_id} not found or doesn't belong to user {user_id}")
        return False
    
    # Check message limit
    current_message_count = len(chat.get("messages", []))
    if current_message_count >= MAX_MESSAGES_PER_CHAT:
        print(f"Chat {chat_id} has reached maximum message limit ({MAX_MESSAGES_PER_CHAT})")
        return False
    
    # Add message metadata
    message_data["timestamp"] = datetime.datetime.now()
    message_data["message_id"] = str(uuid.uuid4())  # Unique message identifier
    message_data["chat_session_id"] = chat.get("chat_session_id")
    
    # Update chat with new message and metadata
    result = chat_collection.update_one(
        {"_id": ObjectId(chat_id), "user_id": user_id},
        {
            "$push": {"messages": message_data},
            "$set": {
                "last_updated": datetime.datetime.now(),
                "chat_metadata.total_messages": current_message_count + 1,
                "chat_metadata.last_message_at": datetime.datetime.now()
            }
        }
    )
    
    success = result.modified_count > 0
    if success:
        print(f"Added message to chat {chat_id} (session: {chat.get('chat_session_id')})")
    
    return success

def delete_chat(chat_id, user_id):
    """Deletes a specific chat for a user."""
    if db is None:
        return False
    
    chat_collection = db.user_chats
    result = chat_collection.delete_one({"_id": ObjectId(chat_id), "user_id": user_id})
    return result.deleted_count > 0

def validate_user_session(user_id, chat_id=None):
    """Validates user session and chat session if provided."""
    if db is None:
        return False, "Database not initialized"
    
    # Validate user session exists and is active
    user_collection = db.user_sessions
    user_session = user_collection.find_one({"user_id": user_id})
    
    if not user_session:
        # If no session exists, that's okay - it will be created
        print(f"No existing session found for user {user_id}, will create new session")
        return True, "Session will be created"
    
    # Check session timeout
    timeout_threshold = datetime.datetime.now() - datetime.timedelta(hours=SESSION_TIMEOUT_HOURS)
    if user_session.get("last_updated", datetime.datetime.min) < timeout_threshold:
        print(f"Session expired for user {user_id}, will create new session")
        # Delete expired session and allow creation of new one
        user_collection.delete_one({"user_id": user_id})
        return True, "Expired session removed, will create new session"
    
    # If chat_id provided, validate chat session
    if chat_id:
        chat_collection = db.user_chats
        chat = chat_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
        
        if not chat:
            return False, "Chat session not found or doesn't belong to user"
        
        # Validate chat session is linked to user session
        if chat.get("user_session_id") != user_session.get("session_id"):
            return False, "Chat session not linked to current user session"
    
    return True, "Session valid"

def rename_chat(chat_id, user_id, new_name):
    """Renames a chat session."""
    if db is None:
        return False
    
    # Validate chat session only
    is_valid, message = validate_user_session(user_id, chat_id)
    if not is_valid:
        print(f"Session validation failed: {message}")
        return False
    
    chat_collection = db.user_chats
    result = chat_collection.update_one(
        {"_id": ObjectId(chat_id), "user_id": user_id},
        {"$set": {"chat_name": new_name, "last_updated": datetime.datetime.now()}}
    )
    
    success = result.modified_count > 0
    if success:
        print(f"Renamed chat {chat_id} to '{new_name}'")
    
    return success

# --- Flask Routes ---

@app.route('/api/chat', methods=['POST'])
def chat():
    start_time = time.time()
    try:
        data = request.json
        user_id = data.get('user_id', str(uuid.uuid4())) # Use a unique ID for each user
        user_message_text = data.get('message')
        user_image_data = data.get('image')
        image_mime_type = data.get('image_mime_type')
        chat_id = data.get('chat_id')

        # First, get or create user session
        session_data = get_or_create_user_session(user_id)
        current_exchange_rate = EXCHANGE_RATE

        # Then validate session and chat if chat_id is provided
        if chat_id:
            is_valid, message = validate_user_session(user_id, chat_id)
            if not is_valid:
                return jsonify({"error": f"Session validation failed: {message}"}), 401

        current_user_message_parts = []

        # --- Constructing the message sent to Gemini ---
        effective_user_message_content = ""

        # Check if user is asking to show all selected products
        show_all_products = False
        show_selected_products = False
        
        # Check for "show all products" requests
        if user_message_text and any(keyword in user_message_text.lower() for keyword in ['show all', 'display all']):
            show_all_products = True
        
        # Check for "show selected products" requests
        if user_message_text and any(keyword in user_message_text.lower() for keyword in ['show selected', 'display selected', 'show my', 'display my', 'products i selected', 'products i chose']):
            show_selected_products = True
            
            # If asking to show all products, add context about previously selected products
            selected_products_from_history = []
            
            # Use chat-specific history if chat_id is provided, otherwise use session history
            chat_id = data.get('chat_id')
            if chat_id:
                # Get products from the specific chat
                chat_messages = get_chat_messages(chat_id, user_id)
                for msg in chat_messages:
                    if msg.get('role') == 'model' and msg.get('type') == 'model_response':
                        # Use new format with product IDs
                        if msg.get('main_recommended_product_ids'):
                            selected_products_from_history.extend(msg['main_recommended_product_ids'])
                        if msg.get('suggestive_recommended_product_ids'):
                            selected_products_from_history.extend(msg['suggestive_recommended_product_ids'])
                        # Fallback to old format if new format not available
                        elif msg.get('main_recommended_products'):
                            selected_products_from_history.extend([p['id'] for p in msg['main_recommended_products']])
                        elif msg.get('suggestive_recommended_products'):
                            selected_products_from_history.extend([p['id'] for p in msg['suggestive_recommended_products']])
            else:
                # Fallback to session history for backward compatibility
                for msg in session_data.get('history', []):
                    if msg.get('role') == 'model' and msg.get('type') == 'model_response':
                        if msg.get('main_recommended_products'):
                            selected_products_from_history.extend([p['id'] for p in msg['main_recommended_products']])
                        if msg.get('suggestive_recommended_products'):
                            selected_products_from_history.extend([p['id'] for p in msg['suggestive_recommended_products']])
            
            if selected_products_from_history:
                # Remove duplicates while preserving order
                unique_selected_products = list(dict.fromkeys(selected_products_from_history))
                effective_user_message_content += f"\n\nPreviously selected products in this conversation: {', '.join(unique_selected_products)}"
                effective_user_message_content += "\n\nPlease show all these previously selected products to the user."

        
        # 2. Add the user's explicit query
        if user_message_text:
            effective_user_message_content += f"User Query: {user_message_text}"
        else:
            # If no user message and no products, prevent empty message
            effective_user_message_content = "Hello! How can I assist you today?"
            print("No user message, setting generic initial prompt.")

        if effective_user_message_content:
            current_user_message_parts.append(glm.Part(text=effective_user_message_content))
        if user_image_data and image_mime_type:
            try:
                image_bytes = base64.b64decode(user_image_data)
                current_user_message_parts.append(glm.Part(inline_data=glm.Blob(mime_type=image_mime_type, data=image_bytes)))
            except Exception as e:
                print(f"Error decoding image: {e}")
                return jsonify({"error": "Invalid image data."}), 400

        # Construct convo_history for Gemini
        convo_history = []
        
        # Use chat-specific history if chat_id is provided, otherwise use session history
        chat_id = data.get('chat_id')
        if chat_id:
            # Get chat-specific history for AI context
            chat_messages = get_chat_messages(chat_id, user_id)
            # Convert chat messages to the format expected by _get_gemini_history
            chat_history_data = {"history": chat_messages}
            convo_history.extend(_get_gemini_history(chat_history_data))
        else:
            # Fallback to session history for backward compatibility
            convo_history.extend(_get_gemini_history(session_data))

        # Get AI response
        ai_response_text, input_tokens, output_tokens = get_gemini_response(
            convo_history,
            current_user_message_parts,
            model_name='gemini-2.5-flash' 
        )

        # Extract potential product IDs from AI's response using the new structured format
        extracted_product_ids_dict = extract_product_ids_from_response(ai_response_text)
        
        main_ids_raw = extracted_product_ids_dict.get("main", [])
        suggestive_ids_raw = extracted_product_ids_dict.get("suggestive", [])

        # Filter for actual products that exist in your catalog
        valid_main_ids = [pid for pid in main_ids_raw if pid in product_catalog]
        valid_suggestive_ids = [pid for pid in suggestive_ids_raw if pid in product_catalog]

            # Normal limits for regular recommendations
        MAX_MAIN_PRODUCTS = 6
        MAX_SUGGESTIVE_PRODUCTS = 6
        final_main_ids = valid_main_ids[:MAX_MAIN_PRODUCTS]
        final_suggestive_ids = valid_suggestive_ids[:MAX_SUGGESTIVE_PRODUCTS]

        # Get full product data for frontend
        main_products_for_frontend = get_recommended_products_data(final_main_ids)
        suggestive_products_for_frontend = get_recommended_products_data(final_suggestive_ids)

        # Calculate costs for the current turn
        turn_costs = get_token_costs(input_tokens, output_tokens, current_exchange_rate)
        if user_image_data:
            image_cost_info = calculate_image_cost(1, current_exchange_rate)
            turn_costs.update(image_cost_info)

        # Update session history and total costs
        # Add user message to history
        user_message_history_entry = {
            "role": "user",
            "type": "text", # Default to text
            "text": user_message_text,
            "timestamp": datetime.datetime.now().isoformat()
        }
        if user_image_data:
            user_message_history_entry["type"] = "image"
            user_message_history_entry["image"] = {
                "data": user_image_data,
                "mime_type": image_mime_type
            }

        session_data['history'].append(user_message_history_entry)

        # If chat_id provided, store messages to that chat and rename first time
        if chat_id:
            try:
                # On first user message, if chat name is default, rename to first prompt (truncated)
                chat_collection = db.user_chats
                chat_doc = chat_collection.find_one({"_id": ObjectId(chat_id), "user_id": user_id})
                if chat_doc:
                    if chat_doc.get('chat_name', '').lower().strip() in ('new chat', 'new chat') and user_message_text:
                        first_prompt_title = user_message_text.strip()[:60]
                        chat_collection.update_one(
                            {"_id": ObjectId(chat_id), "user_id": user_id},
                            {"$set": {"chat_name": first_prompt_title, "last_updated": datetime.datetime.now()}}
                        )

                    # Persist user message
                    add_message_to_chat(chat_id, user_id, {
                        "role": "user",
                        "type": user_message_history_entry["type"],
                        "text": user_message_text,
                        "image": user_message_history_entry.get("image"),
                        "timestamp": datetime.datetime.now()
                    })
            except Exception as persist_e:
                print(f"⚠️ Could not persist user message to chat: {persist_e}")

        # Remove the PRODUCTS_LIST line from the AI response before sending to frontend
        cleaned_ai_response_text = re.sub(r'\n?PRODUCTS_LIST:\s*\{.*?\}', '', ai_response_text, flags=re.DOTALL).strip()

        # --- Constructing the response text for frontend ---
        if show_selected_products:
            # When showing selected products, use a simple message
            total_products = len(main_products_for_frontend) + len(suggestive_products_for_frontend)
            if total_products > 0:
                final_display_text = f"Here are your {total_products} selected products:"
            else:
                final_display_text = "You haven't selected any products yet."
        else:
            # Normal response with suggestive products intro
            suggestive_products_intro_text = ""
            # Only add the intro text if there are actual suggestive products to display
            if suggestive_products_for_frontend:
                 suggestive_products_intro_text = "To perfectly accessorize this outfit, we also have some wonderful complementary items like watches, perfumes, bags, and shoes. These are suggestive products that can go with the recommended outfit."
                 # Add a newline for separation if AI response already has content
                 if cleaned_ai_response_text:
                     suggestive_products_intro_text = "\n\n" + suggestive_products_intro_text
            
            # Combine AI's natural language reply with the backend-managed intro text
            final_display_text = cleaned_ai_response_text + suggestive_products_intro_text

        # Add AI response to history
        # Note: We store the original AI response + displayed products for historical context,
        # but the frontend receives the cleaned text and separate product data.
        # FIXED: Store only product IDs, not full product data to reduce token usage
        session_data['history'].append({
            "role": "model",
            "type": "model_response",
            "text": final_display_text, # Store the final combined text for history
            "main_recommended_product_ids": final_main_ids,  # Store only IDs, not full data
            "suggestive_recommended_product_ids": final_suggestive_ids,  # Store only IDs, not full data
            "cost_info": turn_costs, # Store cost info for this turn
            "timestamp": datetime.datetime.now().isoformat()
        })

        # Also persist model message into chat if chat_id provided
        if chat_id:
            try:
                add_message_to_chat(chat_id, user_id, {
                    "role": "model",
                    "type": "model_response",
                    "text": final_display_text,
                    "main_recommended_product_ids": final_main_ids,  # Store only IDs, not full data
                    "suggestive_recommended_product_ids": final_suggestive_ids,  # Store only IDs, not full data
                    "cost_info": turn_costs,
                    "timestamp": datetime.datetime.now()
                })
            except Exception as persist_e2:
                print(f"⚠️ Could not persist model message to chat: {persist_e2}")

        session_data = update_total_costs(session_data, turn_costs)
        update_user_session(user_id, session_data)

        end_time = time.time()
        print(f"Chat interaction completed in {end_time - start_time:.2f} seconds.")
        print(f"Number of main products sent to frontend: {len(main_products_for_frontend)}")
        print(f"Number of suggestive products sent to frontend: {len(suggestive_products_for_frontend)}")
        print(f"Total products sent to frontend: {len(main_products_for_frontend) + len(suggestive_products_for_frontend)}")

        # Prepare response for frontend
        response_data = {
            "reply": final_display_text, # This will include the AI's natural language + suggestive intro
            "main_recommended_products": main_products_for_frontend,
            "suggestive_recommended_products": suggestive_products_for_frontend,
            "chat_id": chat_id, # Include chat_id in response
            "is_show_all_products": show_all_products, # Flag to indicate if this is a "show all products" request
            "is_show_selected_products": show_selected_products, # Flag to indicate if this is a "show selected products" request
            "conversation_cost": {
                "current_turn_usd": turn_costs.get('total_cost_usd_turn', 0.0),
                "current_turn_pkr": turn_costs.get('total_cost_pkr_turn', 0.0),
                "total_session_usd": session_data['total_cost_usd'],
                "total_session_pkr": session_data['total_cost_pkr'],
                "input_tokens": turn_costs.get('input_tokens', 0),
                "output_tokens": turn_costs.get('output_tokens', 0)
            }
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during chat interaction: {e}")
        traceback.print_exc()
        return jsonify({"reply": "An unexpected error occurred. Please try again later.", "error": str(e)}), 500

@app.route('/api/history', methods=['POST'])
def get_history():
    try:
        data = request.json
        user_id = data.get('user_id', str(uuid.uuid4())) # Use a unique ID for each user
        session_data = get_or_create_user_session(user_id)

        # Filter out sensitive fields or format for client display
        # Here we send the raw history, but in a real app, you might want to sanitize it
        client_history = []
        for msg in session_data.get('history', []):
            # Create a copy to avoid modifying the original session_data
            msg_copy = msg.copy()
            # If there's cost_info, you might want to simplify it or remove it for client
            if 'cost_info' in msg_copy:
                # Optionally, summarize cost_info for the client if needed
                # For now, let's remove it from the client-side history to keep it clean
                del msg_copy['cost_info']
            client_history.append(msg_copy)

        return jsonify({"history": client_history, "total_cost_usd": session_data.get('total_cost_usd', 0.0), "total_cost_pkr": session_data.get('total_cost_pkr', 0.0), "total_input_tokens": session_data.get('total_input_tokens', 0), "total_output_tokens": session_data.get('total_output_tokens', 0), "total_image_displays": session_data.get('total_image_displays', 0)})
    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({"error": "Failed to retrieve history."}), 500

@app.route('/api/product-details/<product_id>', methods=['GET'])
def get_product_details(product_id):
    """
    Retrieves full details for a single product from the product_catalog.
    """
    product = product_catalog.get(product_id)
    if product:
        return jsonify(product)
    else:
        return jsonify({"error": "Product not found."}), 404

@app.route('/api/product-analysis', methods=['POST'])
def analyze_product():
    """
    Analyzes a product using the Gemini model and returns the analysis.
    Expects a JSON body with 'product_id' and 'user_id'.
    """
    try:
        data = request.json
        product_id = data.get('product_id')
        user_id = data.get('user_id', str(uuid.uuid4())) # Default user_id

        if not product_id:
            return jsonify({"error": "Product ID is required for analysis."}), 400

        product_details_for_ai = get_product_details_for_ai(product_id)
        if "not found" in product_details_for_ai.lower():
            return jsonify({"error": product_details_for_ai}), 404

        # Prepare a prompt for Gemini for product analysis
        analysis_prompt = f"Analyze the following product details. Provide a comprehensive description, style recommendations, and suitable occasions for wearing/using this product. Focus on its features, materials, and overall aesthetic. Product Details:\n\n{product_details_for_ai}"

        # Get user session to maintain history for analysis context
        session_data = get_or_create_user_session(user_id)
        current_exchange_rate = EXCHANGE_RATE

        # --- ACTUAL GEMINI API CALL ---
        print(f"Calling Gemini for analysis of product ID: {product_id}")
        print(f"Analysis Prompt: {analysis_prompt}")

        ai_response_text = "" # Initialize to empty string
        try:
            # Make the actual call to the Gemini model
            response = gemini_model.generate_content(analysis_prompt)
            ai_response_text = response.text # Extract the text content from the response
            clean_response = re.sub(r'\n?PRODUCTS_LIST:\s*\{.*?\}', '', ai_response_text, flags=re.DOTALL).strip()


            print(f"Gemini Raw Response: {response}") # For debugging the full Gemini response object
            print(f"Gemini Generated Text: {clean_response[:200]}...") # Print first 200 chars for brevity

        except genai.types.BlockedPromptException as e:
            # Handle cases where Gemini might block content (e.g., safety reasons)
            print(f"Gemini Blocked Prompt: {e}")
            ai_response_text = "I apologize, but I couldn't generate an analysis due to content policy. Please try a different product or rephrase your request."
            # You might want to return a different status code or error message to the frontend here
            # return jsonify({"error": f"AI analysis blocked: {str(e)}"}), 400
        except Exception as gemini_e:
            # Catch other potential errors from the Gemini API call (e.g., API key issues, network)
            print(f"Error calling Gemini API: {gemini_e}")
            traceback.print_exc() # Print full traceback for debugging
            ai_response_text = "I apologize, but I couldn't generate an analysis for this product at the moment. Please try again or provide more details."
            # You might want to return a different status code or error message to the frontend here
            # return jsonify({"error": f"Failed to get AI analysis: {str(gemini_e)}"}), 500


        # Calculate token costs using the actual generated response
        simulated_input_tokens = len(analysis_prompt.split()) # Simple word count as token estimate
        simulated_output_tokens = len(ai_response_text.split()) # Use actual AI response text for output tokens

        turn_costs = get_token_costs(simulated_input_tokens, simulated_output_tokens, current_exchange_rate)
        
        # Optionally, record this analysis interaction in the user's history
        session_data['history'].append({
            "role": "user",
            "type": "text",
            "text": f"Analyze product with ID: {product_id}",
            "timestamp": datetime.datetime.now().isoformat()
        })
        session_data['history'].append({
            "role": "model",
            "type": "model_response",
            "text": clean_response, # Use the actual AI response here
            "main_recommended_products": [product_catalog[product_id]] if product_id in product_catalog else [], # Store as main
            "suggestive_recommended_products": [], # No suggestive products for direct analysis
            "cost_info": turn_costs,
            "timestamp": datetime.datetime.now().isoformat()
        })
        session_data = update_total_costs(session_data, turn_costs)
        update_user_session(user_id, session_data)

        # --- RETURN ACTUAL AI RESPONSE ---
        return jsonify({
            "analysis_result":clean_response, # <--- Sending the actual AI response here
            "product_details": product_catalog[product_id], # Keep sending this if frontend uses it
            "conversation_cost": {
                "current_turn_usd": turn_costs.get('total_cost_usd_turn', 0.0),
                "current_turn_pkr": turn_costs.get('total_cost_pkr_turn', 0.0),
                "total_session_usd": session_data['total_cost_usd'],
                "total_session_pkr": session_data['total_cost_pkr'],
                "input_tokens": turn_costs.get('input_tokens', 0),
                "output_tokens": turn_costs.get('output_tokens', 0)
            }
        })

    except Exception as e:
        print(f"Error during product analysis: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred during analysis: {str(e)}"}), 500

@app.route('/api/show-all-products', methods=['POST'])
def show_all_products():
    """Shows all products that were previously recommended in the conversation."""
    try:
        data = request.json
        user_id = data.get('user_id')
        chat_id = data.get('chat_id')
        
        if not user_id:
            return jsonify({"error": "User ID is required."}), 400
        
        # If chat_id is provided, get products from that specific chat
        if chat_id:
            chat_messages = get_chat_messages(chat_id, user_id)
            all_products = []
            product_ids_seen = set()
            
            for msg in chat_messages:
                if msg.get('role') == 'model' and msg.get('type') == 'model_response':
                    # Add main products
                    if msg.get('main_recommended_products'):
                        for product in msg['main_recommended_products']:
                            if product.get('id') and product['id'] not in product_ids_seen:
                                all_products.append(product)
                                product_ids_seen.add(product['id'])
                    
                    # Add suggestive products
                    if msg.get('suggestive_recommended_products'):
                        for product in msg['suggestive_recommended_products']:
                            if product.get('id') and product['id'] not in product_ids_seen:
                                all_products.append(product)
                                product_ids_seen.add(product['id'])
        else:
            # Fallback to session-based history for backward compatibility
            session_data = get_or_create_user_session(user_id)
            all_products = []
            product_ids_seen = set()
            
            for msg in session_data.get('history', []):
                if msg.get('role') == 'model' and msg.get('type') == 'model_response':
                    # Add main products
                    if msg.get('main_recommended_products'):
                        for product in msg['main_recommended_products']:
                            if product.get('id') and product['id'] not in product_ids_seen:
                                all_products.append(product)
                                product_ids_seen.add(product['id'])
                    
                    # Add suggestive products
                    if msg.get('suggestive_recommended_products'):
                        for product in msg['suggestive_recommended_products']:
                            if product.get('id') and product['id'] not in product_ids_seen:
                                all_products.append(product)
                                product_ids_seen.add(product['id'])
        
        # When user asks to "show all products", separate products into main and suggestive
        main_products = []
        suggestive_products = []
        
        # Separate products based on their original categorization
        if chat_id:
            chat_messages = get_chat_messages(chat_id, user_id)
            main_product_ids_seen = set()
            suggestive_product_ids_seen = set()
            
            for msg in chat_messages:
                if msg.get('role') == 'model' and msg.get('type') == 'model_response':
                    # Add main products
                    if msg.get('main_recommended_products'):
                        for product in msg['main_recommended_products']:
                            if product.get('id') and product['id'] not in main_product_ids_seen:
                                main_products.append(product)
                                main_product_ids_seen.add(product['id'])
                    
                    # Add suggestive products
                    if msg.get('suggestive_recommended_products'):
                        for product in msg['suggestive_recommended_products']:
                            if product.get('id') and product['id'] not in suggestive_product_ids_seen:
                                suggestive_products.append(product)
                                suggestive_product_ids_seen.add(product['id'])
        else:
            # Fallback to session-based history for backward compatibility
            session_data = get_or_create_user_session(user_id)
            main_product_ids_seen = set()
            suggestive_product_ids_seen = set()
            
            for msg in session_data.get('history', []):
                if msg.get('role') == 'model' and msg.get('type') == 'model_response':
                    # Add main products
                    if msg.get('main_recommended_products'):
                        for product in msg['main_recommended_products']:
                            if product.get('id') and product['id'] not in main_product_ids_seen:
                                main_products.append(product)
                                main_product_ids_seen.add(product['id'])
                    
                    # Add suggestive products
                    if msg.get('suggestive_recommended_products'):
                        for product in msg['suggestive_recommended_products']:
                            if product.get('id') and product['id'] not in suggestive_product_ids_seen:
                                suggestive_products.append(product)
                                suggestive_product_ids_seen.add(product['id'])
        
        total_products = len(main_products) + len(suggestive_products)
        reply_text = f"I'm showing you all {total_products} products that were recommended in our conversation. Here they are:"
        
        if len(main_products) > 0 and len(suggestive_products) > 0:
            reply_text += f"\n\nMain Products ({len(main_products)} items):"
            reply_text += f"\nAccessories ({len(suggestive_products)} items):"
        elif len(main_products) > 0:
            reply_text += f"\n\nMain Products ({len(main_products)} items):"
        elif len(suggestive_products) > 0:
            reply_text += f"\n\nAccessories ({len(suggestive_products)} items):"
        
        return jsonify({
            "reply": reply_text,
            "main_recommended_products": main_products,
            "suggestive_recommended_products": suggestive_products,
            "is_show_all_products": True, # Flag to indicate this is a "show all products" request
            "total_products": total_products
        })
        
    except Exception as e:
        print(f"Error showing all products: {e}")
        return jsonify({"error": "Failed to retrieve all products."}), 500

# --- Multi-Chat API Endpoints ---

@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Creates a new chat session."""
    try:
        data = request.json
        user_id = data.get('user_id')
        chat_name = data.get('chat_name', 'New Chat')
        
        if not user_id:
            return jsonify({"error": "User ID is required."}), 400
        
        # Create user session if it doesn't exist
        session_data = get_or_create_user_session(user_id)
        
        # Create new chat
        new_chat = create_new_chat(user_id, chat_name)
        if new_chat:
            return jsonify({"success": True, "chat": new_chat}), 201
        else:
            return jsonify({"error": "Failed to create chat. User may have reached chat limit."}), 500
            
    except Exception as e:
        print(f"Error creating chat: {e}")
        return jsonify({"error": "Failed to create chat."}), 500

@app.route('/api/chats/<user_id>', methods=['GET'])
def get_chats(user_id):
    """Retrieves all chat sessions for a user."""
    try:
        # Get or create user session
        session_data = get_or_create_user_session(user_id)
        
        chats = get_user_chats(user_id)
        return jsonify({"chats": chats}), 200
    except Exception as e:
        print(f"Error getting chats: {e}")
        return jsonify({"error": "Failed to retrieve chats."}), 500

@app.route('/api/chats/<chat_id>/messages', methods=['GET', 'POST'])
def get_messages(chat_id):
    """Retrieves messages for a specific chat."""
    try:
        user_id = None
        # Support both GET (query param) and POST (json body)
        if request.method == 'GET':
            user_id = request.args.get('user_id')
        else:
            data = request.get_json(silent=True) or {}
            user_id = data.get('user_id')

        if not user_id:
            return jsonify({"error": "User ID is required."}), 400

        # Get or create user session
        session_data = get_or_create_user_session(user_id)

        # Validate chat session only
        is_valid, message = validate_user_session(user_id, chat_id)
        if not is_valid:
            return jsonify({"error": f"Session validation failed: {message}"}), 401

        messages = get_chat_messages(chat_id, user_id)
        
        # Format messages for frontend
        formatted_messages = []
        for msg in messages:
            # Function to map product data for frontend
            def map_products_for_frontend(products):
                if not products:
                    return []
                return products  # Products are already in the correct format from the database
            
            # Determine message type based on content
            has_products = (msg.get('main_recommended_products') and len(msg.get('main_recommended_products', [])) > 0) or \
                          (msg.get('suggestive_recommended_products') and len(msg.get('suggestive_recommended_products', [])) > 0)
            
            formatted_msg = {
                "role": msg.get('role'),
                "type": "productDisplay" if has_products else "text",
                "text": msg.get('text'),
                "timestamp": msg.get('timestamp'),
                "mainProducts": map_products_for_frontend(msg.get('main_recommended_products', [])),
                "suggestiveProducts": map_products_for_frontend(msg.get('suggestive_recommended_products', []))
            }
            
            # Handle image data if present
            if msg.get('image'):
                formatted_msg['image'] = {
                    "data": msg['image'].get('data'),
                    "mime_type": msg['image'].get('mime_type')
                }
            
            formatted_messages.append(formatted_msg)
        
        return jsonify({"messages": formatted_messages}), 200
    except Exception as e:
        print(f"Error getting messages: {e}")
        return jsonify({"error": "Failed to retrieve messages."}), 500

@app.route('/api/chats/<chat_id>/rename', methods=['PUT'])
def rename_chat_endpoint(chat_id):
    """Renames a chat session."""
    try:
        data = request.json
        user_id = data.get('user_id')
        new_name = data.get('chat_name')
        
        if not user_id or not new_name:
            return jsonify({"error": "User ID and chat name are required."}), 400
        
        # Get or create user session
        session_data = get_or_create_user_session(user_id)
        
        success = rename_chat(chat_id, user_id, new_name)
        if success:
            return jsonify({"success": True, "message": "Chat renamed successfully."}), 200
        else:
            return jsonify({"error": "Failed to rename chat."}), 500
            
    except Exception as e:
        print(f"Error renaming chat: {e}")
        return jsonify({"error": "Failed to rename chat."}), 500

@app.route('/api/session/info', methods=['POST'])
def get_session_info():
    """Gets information about the current user session."""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({"error": "User ID is required."}), 400
        
        # Get or create user session
        session_data = get_or_create_user_session(user_id)
        
        # Get user session data
        user_collection = db.user_sessions
        user_session = user_collection.find_one({"user_id": user_id})
        
        if not user_session:
            return jsonify({"error": "User session not found."}), 404
        
        # Get user's chats
        chat_collection = db.user_chats
        user_chats = list(chat_collection.find({"user_id": user_id}).sort("last_updated", -1))
        
        # Convert ObjectIds to strings
        for chat in user_chats:
            chat["_id"] = str(chat["_id"])
        
        session_info = {
            "user_id": user_id,
            "session_id": user_session.get("session_id"),
            "created_at": user_session.get("created_at").isoformat() if user_session.get("created_at") else None,
            "last_updated": user_session.get("last_updated").isoformat() if user_session.get("last_updated") else None,
            "total_chats": len(user_chats),
            "session_metadata": user_session.get("session_metadata", {}),
            "costs": {
                "total_cost_usd": user_session.get("total_cost_usd", 0.0),
                "total_cost_pkr": user_session.get("total_cost_pkr", 0.0),
                "total_input_tokens": user_session.get("total_input_tokens", 0),
                "total_output_tokens": user_session.get("total_output_tokens", 0),
                "total_image_displays": user_session.get("total_image_displays", 0)
            }
        }
        
        return jsonify({"success": True, "session_info": session_info}), 200
        
    except Exception as e:
        print(f"Error getting session info: {e}")
        return jsonify({"error": "Failed to get session info."}), 500

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat_endpoint(chat_id):
    """Deletes a chat session."""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({"error": "User ID is required."}), 400
        
        # Get or create user session
        session_data = get_or_create_user_session(user_id)
        
        # Validate chat session only
        is_valid, message = validate_user_session(user_id, chat_id)
        if not is_valid:
            return jsonify({"error": f"Session validation failed: {message}"}), 401
        
        success = delete_chat(chat_id, user_id)
        if success:
            return jsonify({"success": True, "message": "Chat deleted successfully."}), 200
        else:
            return jsonify({"error": "Failed to delete chat."}), 500
            
    except Exception as e:
        print(f"Error deleting chat: {e}")
        return jsonify({"error": "Failed to delete chat."}), 500

def reset_all_user_sessions():
    """
    Resets all user sessions to clear problematic history data.
    This is a drastic measure to fix token usage issues.
    """
    if db is None:
        print("MongoDB not initialized, cannot reset sessions.")
        return False
    
    user_collection = db.user_sessions
    result = user_collection.delete_many({})
    print(f"Reset {result.deleted_count} user sessions to clear problematic history.")
    return True

@app.route('/api/reset-sessions', methods=['POST'])
def reset_sessions_endpoint():
    """Resets all user sessions to clear problematic history data."""
    try:
        success = reset_all_user_sessions()
        if success:
            return jsonify({"success": True, "message": "All user sessions reset successfully."}), 200
        else:
            return jsonify({"error": "Failed to reset sessions."}), 500
    except Exception as e:
        print(f"Error resetting sessions: {e}")
        return jsonify({"error": "Failed to reset sessions."}), 500

if __name__ == '__main__':
    # Disable the reloader to avoid Windows/Python 3.13 shutdown thread issues
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)