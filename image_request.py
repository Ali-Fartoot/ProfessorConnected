import requests
import base64
import os
from PIL import Image
import io

# Make the request
response = requests.post(
    "http://localhost:8000/search_with_visualization",
    json={"text_query": "Sergey Levine", "limit": 5, "min_similarity": 0.1}
)

# Get the base64 string
base64_string = response.json()["network_image"]

# Convert base64 to image and save
img_data = base64.b64decode(base64_string)
image = Image.open(io.BytesIO(img_data))
image.save("visualization.png")
