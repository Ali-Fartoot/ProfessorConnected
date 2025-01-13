from crawler import crawl
from pdf_to_text import AuthorDocumentProcessor
import os
import time
from vector_search import cleanup_database, add_professor, find_similar_professor

start = time.time()

name = "Majid Nili Ahmadabadi"
data_path = os.path.join("data", name)
json_path = os.path.join(data_path, name + ".json")

os.makedirs("data", exist_ok=True)

# Step 1: Crawl papers
if not os.path.exists(data_path):
    crawl(name, number_of_articles=2)
    print("Crawling completed.")
else:
    print("The author papers already exist.")

# Step 2: Process documents
if os.path.exists(data_path) and not os.path.exists(json_path):
    document_processor = AuthorDocumentProcessor()
    document_processor(name)
    print("Document processing completed.")
else:
    print("The author papers are already processed or crawling failed.")

# Step 3: Add professor to database
if os.path.exists(json_path):
    add_professor(name)
    print("Professor added to database.")
else:
    print("Document processing failed, professor not added to database.")

# Step 4: Cleanup database
cleanup_database()
print("Database cleanup completed.")

end = time.time()
print(f"The app took {end - start} seconds.")