from crawler import crawl
from pdf_to_text import AuthorDocumentProcessor
import os
import time

start = time.time()

name = "Majid Nili Ahmadabadi"

os.makedirs("data", exist_ok=True)
if not os.path.exists(os.path.join("data", name)):
    crawl(name, number_of_articles=2)
else:
    print("The author papers already exists")


document_processor = AuthorDocumentProcessor()
if not os.path.exists(os.path.join("data", name, name + ".json")):
    document_processor(name)
else:
    print("The author papers are already processed")

end = time.time()

print(f"The app took {end - start}")