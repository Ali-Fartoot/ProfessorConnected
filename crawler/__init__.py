import arxivpy
from crawler.utils import convert_to_arxiv_query
import os

def crawl(name: str, number_of_articles: int = 10) -> None:
    search_query = convert_to_arxiv_query(name)
    articles = arxivpy.query(search_query=search_query,wait_time=3.0, sort_by='lastUpdatedDate')[:number_of_articles]
    arxivpy.download(articles, path=f'data/{name}')
    
    for article in articles:
        # Get the article's ID and title
        article_id = article["id"]
        article_title = article["title"]
        
        # Sanitize the title to make it a valid filename
        sanitized_title = "".join([c if c.isalnum() or c in " -_" else "_" for c in article_title])
        
        # Construct the old and new file paths
        old_file_path = os.path.join(f'data/{name}', f"{article_id}.pdf")
        new_file_path = os.path.join(f'data/{name}', f"{sanitized_title}.pdf")
        
        # Rename the file
        if os.path.exists(old_file_path):
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {article_id}.pdf -> {sanitized_title}.pdf")
        else:
            print(f"File not found: {old_file_path}")