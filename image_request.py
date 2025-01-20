import requests
import base64
import os
from PIL import Image
import io
import json

# Make the request
response = requests.post(
    "http://localhost:8000/search_with_visualization",
    json={"text_query": "Sergey Levine", "limit": 5, "min_similarity": 0.1}
)

# Print the response for debugging
print("Response Status Code:", response.status_code)
print("Response Content:", response.text)

try:
    # Get the base64 string
    response_json = response.json()
    print("Parsed JSON:", json.dumps(response_json, indent=2))
    
    base64_string = response_json["network_image"]

    # Convert base64 to image and save
    img_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_data))
    image.save("visualization.png")
except KeyError as e:
    print(f"KeyError: The key '{e}' was not found in the response")
except json.JSONDecodeError:
    print("Error: Could not parse JSON response")
except Exception as e:
    print(f"Error: {str(e)}")