import os
from dotenv import load_dotenv
import google.genai as genai

# Load environment variables from .env
load_dotenv()

# Initialize the Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
client = genai.Client(api_key=api_key)


def get_investment_advice(predictions, stock_symbol):
    """
    Generate investment advice using Gemini API based on predicted stock prices.
    """
    prompt = (
        f"Stock symbol: {stock_symbol}\n"
        f"Predicted prices for the next days: {predictions}\n"
        "Provide stock analysis, insights to the user, "
        "and suggest whether to buy, hold, or sell with reasoning."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip() if response.text else "No advice generated."
    except Exception as e:
        print("Error generating investment advice:", e)
        return "Could not generate advice at this time."
