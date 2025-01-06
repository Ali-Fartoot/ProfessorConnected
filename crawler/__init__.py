import arxivpy

search_query='au:Majid+AND+au:Nili'
articles = arxivpy.query(search_query=search_query,wait_time=5.0, sort_by='lastUpdatedDate')
arxivpy.download(articles, path='arxiv_pdf')
