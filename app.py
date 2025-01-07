from crawler import crawl
from pdf_to_text import AuthorDocumentProcessor
import os

document_processor = AuthorDocumentProcessor()
name = input()
if not os.path.exists(os.path.join("data", name)):
    crawl(name)
else:
    print("The author papers already exists")

if not os.path.exists(os.path.join("data", name, name + ".json")):
    document_processor(name)
else:
    print("The author papers are already processed")