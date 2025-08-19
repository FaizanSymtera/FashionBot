# backendCostCalculation.py

# --- Gemini Model Costs (as of early 2024, subject to change by Google) ---
# For gemini-2.5-flash (text-only)
# Input tokens cost per 1000 tokens in USD
TOKEN_COST_INPUT_FLASH = 0.000125
# Output tokens cost per 1000 tokens in USD
TOKEN_COST_OUTPUT_FLASH = 0.000375

# For gemini-2.5-flash (multimodal, for image analysis)
# Input tokens cost per 1000 tokens in USD
TOKEN_COST_INPUT_VISION = 0.000125
# Output tokens cost per 1000 tokens in USD
TOKEN_COST_OUTPUT_VISION = 0.000375

# Image cost per image (e.g., for 1080p equivalent) in USD
IMAGE_COST_VISION = 0.0025 # Example cost for an image, adjust based on actual pricing

# --- Convenience constants for easier access ---
# These are the ones your NewBackend.py is likely trying to import
TOKEN_COST_MODEL = {
    "gemini-2.5-flash": {
        "input": TOKEN_COST_INPUT_FLASH,
        "output": TOKEN_COST_OUTPUT_FLASH
    }
}

TOKEN_COST_VISION = { # Specific costs for vision model if needed separately
    "input": TOKEN_COST_INPUT_VISION,
    "output": TOKEN_COST_OUTPUT_VISION,
    "image": IMAGE_COST_VISION
}

# --- Exchange Rate (will be fetched dynamically in NewBackend.py) ---
# This file doesn't need to define the EXCHANGE_RATE itself, as NewBackend.py fetches it.
# Just ensuring that the backend will use this file for constants.

# You can add more detailed cost logic here if needed,
# e.g., for different image resolutions or specific API calls.