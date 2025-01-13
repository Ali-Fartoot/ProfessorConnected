import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import networkx as nx
from typing import List, Dict
import pandas as pd
from vector_search import ProfessorResearchProfile

class ProfessorVisualizer:
    def __init__(self, profile_system: ProfessorResearchProfile):
        """
        Initialize visualizer with ProfessorResearchProfile system
        """
        self.profile_system = profile_system
        self.colors = px.colors.qualitative.Set3
        
    def _get_all_professors_embeddings(self):
        """Get embeddings for all professors"""
        all_points = self.profile_system.client.scroll(
            collection_name=self.profile_system.collection_name,
            limit=1000  # Adjust based on your needs
        )[0]
        
        embeddings = []
        names = []
        departments = []
        universities = []
        
        for point in all_points:
            embeddings.append(point.vector)
            names.append(point.payload['name'])
            departments.append(point.payload.get('department', 'Unknown'))
            universities.append(point.payload.get('university', 'Unknown'))
            
        return np.array(embeddings), names, departments, universities

    def create_network_graph(self, similarity_threshold: float = 0.7):
        """
        Create an interactive network graph showing professor relationships
        """
        embeddings, names, departments, universities = self._get_all_professors_embeddings()
        
        # Calculate similarity matrix
        similarity_matrix = np.inner(embeddings, embeddings)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(names):
            G.add_node(name, 
                      department=departments[i],
                      university=universities[i])
        
        # Add edges based on similarity
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if similarity_matrix[i][j] > similarity_threshold:
                    G.add_edge(names[i], names[j], 
                             weight=similarity_matrix[i][j])
        
        # Get position layout
        pos = nx.spring_layout(G)
        
        # Create edges trace
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G.edges[edge]['weight'])
        
        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create nodes trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(
                f"Professor: {node}<br>"
                f"Department: {G.nodes[node]['department']}<br>"
                f"University: {G.nodes[node]['university']}"
            )
            # Assign colors based on department
            node_colors.append(hash(G.nodes[node]['department']) % len(self.colors))
        
        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color=[self.colors[c] for c in node_colors],
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=[edges_trace, nodes_trace],
                       layout=go.Layout(
                           title='Professor Research Similarity Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig

    def create_2d_projection(self):
        """
        Create 2D projection of professor similarities using t-SNE
        """
        embeddings, names, departments, universities = self._get_all_professors_embeddings()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'Professor': names,
            'Department': departments,
            'University': universities
        })
        
        # Create scatter plot
        fig = px.scatter(df, x='x', y='y',
                        color='Department',
                        hover_data=['Professor', 'University'],
                        title='2D Projection of Professor Research Similarities')
        
        fig.update_traces(marker=dict(size=10))
        
        return fig

    def create_similarity_heatmap(self, professor_name: str, top_n: int = 10):
        """
        Create a heatmap showing similarities between a professor and their top N most similar colleagues
        """
        # Get similar professors
        similar_profs = self.profile_system.find_similar_professors(
            professor_name=professor_name,
            limit=top_n,
            min_similarity=0.0
        )
        
        # Create similarity matrix
        names = [professor_name] + [prof['name'] for prof in similar_profs]
        
        similarity_matrix = np.zeros((len(names), len(names)))
        similarity_matrix[0, 1:] = [prof['similarity_score'] for prof in similar_profs]
        similarity_matrix[1:, 0] = similarity_matrix[0, 1:]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=names,
            y=names,
            colorscale='Viridis',
            text=np.round(similarity_matrix, 2),
            texttemplate='%{text}',
            textfont={'size': 10},
            hoverongaps=False))
        
        fig.update_layout(
            title=f'Research Similarity Heatmap for {professor_name}',
            xaxis_title='Professors',
            yaxis_title='Professors'
        )
        
        return fig
    
    def save_figures(self, professor_name: str, output_dir: str = "./figures", 
                    format: str = "png", limit: int = 5):
        """
        Generate and save all visualization figures
        
        Args:
            professor_name (str): Name of the professor
            output_dir (str): Directory to save figures
            format (str): Image format ('png', 'jpg', 'pdf', 'svg')
            limit (int): Number of similar professors to include
        """
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create and save network graph
        network_fig = self.create_network_graph(professor_name, limit=limit)
        network_path = os.path.join(output_dir, 
                                  f"network_{professor_name}_{timestamp}.{format}")
        network_fig.write_image(network_path)
        print(f"Network graph saved to: {network_path}")
        
        # Create and save heatmap
        heatmap_fig = self.create_similarity_heatmap(professor_name, limit=limit)
        heatmap_path = os.path.join(output_dir, 
                                   f"heatmap_{professor_name}_{timestamp}.{format}")
        heatmap_fig.write_image(heatmap_path)
        print(f"Heatmap saved to: {heatmap_path}")
        return heatmap_fig


# Usage Example
if __name__ == "__main__":
    visualizer = ProfessorVisualizer()
    visualizer.save_figures(
        professor_name="Majid Nili Ahmadabadi",
        output_dir="./figures",
        format="png",
        limit=5
    )