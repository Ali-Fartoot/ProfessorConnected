from crawler import crawl
from pdf_to_text import AuthorDocumentProcessor
import os
import time
from vector_search import cleanup_database, add_professor,  find_hybrid_search_professors
from vector_search.visulizer import ProfessorVisualizer
start = time.time()

professor = ["Manuel Cebrian","Mohammad Noorchenarboo","Yongfan Lai",
             "Kai Li", "Tian Lan", "Hani S. Mahmassani", "Sven Klaassen",
             "Kenichi Shimizu", " Andrey Ramos", "Wei Zhao", "Matt Schwartz",
             "Nobutaka Ono", "Bodong Shang", "Meng-Xing Tang", "Geoffrey Ye Li","Ilya Sutskever"
             "Andrew Ng", "Mohammad Abu Tami", "Li Weigang", "David Silver", "Pieter Abbeel", "Sergey Levine"]

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

# Visualization after all professors are processed
visualizer = ProfessorVisualizer()
image = visualizer.save_figures(
    professor_name="Sergey Levine",
    output_dir="data/Sergey Levine",  
    format="png",
    limit=5
)

end = time.time()
print(f"The app took {end - start} seconds.")