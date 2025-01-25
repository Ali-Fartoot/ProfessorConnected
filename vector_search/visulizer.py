import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import networkx as nx
from typing import List, Dict
import pandas as pd
from vector_search import ProfessorResearchProfile
import os, datetime
import json

class ProfessorVisualizer:
    def __init__(self, path: str = "./professor_db"):
        self.path = path
        self.colors = px.colors.qualitative.Set3

    def create_network_graph(self, professor_name: str, limit: int = 5, min_similarity= 0.1):
        with ProfessorResearchProfile(path=self.path) as profile_system:
            similar_profs = profile_system.hybrid_search_by_professor(
                professor_name=professor_name,
                limit=limit,
                min_similarity=min_similarity
            )

        G = nx.Graph()
        G.add_node(professor_name,
                  size=30,
                  color=self.colors[0],
                  keywords=[]) 

        for i, prof in enumerate(similar_profs):
            if prof['name'] != professor_name:
                G.add_node(prof['name'],
                          size=20,
                          color=self.colors[i + 1],
                          keywords=prof['top_keywords'],
                          shared_keywords=prof['shared_top_keywords'])
                
                G.add_edge(professor_name, 
                          prof['name'],
                          weight=prof['combined_score'])

        pos = nx.spring_layout(G, k=2, iterations=50)
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            score = G.edges[edge]['weight']
            edge_text.append(f"Similarity Score: {score:.3f}")

        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == professor_name:
                node_text.append(f"Professor: {node}")
            else:
                shared_keywords = G.nodes[node].get('shared_keywords', [])
                node_text.append(
                    f"Professor: {node}<br>"
                    f"Top Keywords: {', '.join(G.nodes[node]['keywords'])}<br>"
                    f"Shared Keywords: {', '.join(shared_keywords)}<br>"
                    f"Similarity Score: {G.edges[(professor_name, node)]['weight']:.3f}"
                )
            
            node_colors.append(G.nodes[node]['color'])
            node_sizes.append(G.nodes[node]['size'])

        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            textposition="bottom center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=2))

        fig = go.Figure(data=[edges_trace, nodes_trace],
                       layout=go.Layout(
                           title=f'Research Similarity Network for {professor_name}',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=1000,
                           height=800
                       ))

        return fig

    def create_similarity_heatmap(self, professor_name: str, limit: int = 5):
        with ProfessorResearchProfile(path=self.path) as profile_system:
            similar_profs = profile_system.hybrid_search_by_professor(
                professor_name=professor_name,
                limit=limit
            )

        similar_profs = [prof for prof in similar_profs if prof['name'] != professor_name]
        names = [professor_name] + [prof['name'] for prof in similar_profs]
        matrix_size = len(names)
        similarity_matrix = np.zeros((matrix_size, matrix_size))
        
        for i, prof in enumerate(similar_profs):
            similarity_matrix[0, i+1] = prof['combined_score']
            similarity_matrix[i+1, 0] = prof['combined_score']

        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=names,
            y=names,
            colorscale='Viridis',
            text=np.round(similarity_matrix, 3),
            texttemplate='%{text}',
            textfont={'size': 10}))

        fig.update_layout(
            title=f'Research Similarity Heatmap for {professor_name}',
            xaxis_title='Professors',
            yaxis_title='Professors',
            width=800,
            height=800
        )

        return fig

    def save_figures(self, professor_name: str, 
                    output_dir: str = "./figures", 
                    format: str = "png", 
                    limit: int = 5,
                    min_similarity=0.1):

        os.makedirs(output_dir, exist_ok=True)
        
        network_fig = self.create_network_graph(professor_name, limit=limit, min_similarity=min_similarity)
        network_path = os.path.join(output_dir, 
                                  f"network_{professor_name}.{format}")
        network_fig.write_image(network_path)
        print(f"Network graph saved to: {network_path}")
        
        return network_path