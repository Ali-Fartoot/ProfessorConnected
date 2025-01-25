import arxivpy
from crawler.utils import convert_to_arxiv_query
import os

def crawl(name: str, number_of_articles: int = 10) -> None:
    search_query = convert_to_arxiv_query(name)
    articles = arxivpy.query(search_query=search_query,wait_time=3.0, sort_by='lastUpdatedDate')[:number_of_articles]
    if not articles:
        raise ValueError("No articles to download")
    arxivpy.download(articles, path=f'data/{name}')
    for article in articles:
        if not article:
            continue

        article_id = article["id"]
        article_title = article["title"]
        sanitized_title = "".join([c if c.isalnum() or c in " -_" else "_" for c in article_title])
        old_file_path = os.path.join(f'data/{name}', f"{article_id}.pdf")
        new_file_path = os.path.join(f'data/{name}', f"{sanitized_title}.pdf")

        if os.path.exists(old_file_path):
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {article_id}.pdf -> {sanitized_title}.pdf")
        else:
            print(f"File not found: {old_file_path}")