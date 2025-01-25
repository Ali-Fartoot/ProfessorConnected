def convert_to_arxiv_query(name: str) -> str:
    name = ["au:" + n.capitalize() for n in name.split()]
    name = "+AND+".join(name)
    return name

