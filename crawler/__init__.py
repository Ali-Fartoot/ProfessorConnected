import arxivpy
from crawler.utils import convert_to_arxiv_query

def crawl(name: str, number_of_articles: int = 10) -> None:
    search_query = convert_to_arxiv_query(name)
    articles = arxivpy.query(search_query=search_query,wait_time=3.0, sort_by='lastUpdatedDate')[:number_of_articles]
    arxivpy.download(articles, path=f'data/{name}')
    return
