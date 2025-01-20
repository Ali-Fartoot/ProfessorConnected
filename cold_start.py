from crawler import crawl
from pdf_to_text import AuthorDocumentProcessor
import os
import time
from vector_search import add_professor
from vector_search.visulizer import ProfessorVisualizer

professor = ["Manuel Cebrian","Mohammad Noorchenarboo","Yongfan Lai",
             "Kai Li", "Tian Lan", "Hani S. Mahmassani", "Sven Klaassen",
             "Kenichi Shimizu", " Andrey Ramos", "Wei Zhao", "Matt Schwartz",
             "Nobutaka Ono", "Bodong Shang","Andrew Ng", "Li Weigang", "David Silver", "Pieter Abbeel", "Sergey Levine"]

# Process all professors
for name in professor:
    data_path = os.path.join("data", name)
    json_path = os.path.join(data_path, name + ".json")

    os.makedirs("data", exist_ok=True)

    # Step 1: Crawl papers
    if not os.path.exists(data_path):
        crawl(name, number_of_articles=3)
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

