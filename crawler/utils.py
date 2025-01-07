def convert_to_arxiv_query(name: str) -> str:
    """
    Convert a name to a arXiv query string
    """
    name = ["au:" + n.capitalize() for n in name.split()]
    name = "+AND+".join(name)
    return name

