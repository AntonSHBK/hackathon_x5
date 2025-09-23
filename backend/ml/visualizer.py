from typing import Literal, Optional, Dict, Any

import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import cKDTree

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering

class DynamicEmbeddingVisualizer:

    def __init__(self, embeddings: np.ndarray, data_df: pd.DataFrame, labels: np.ndarray = None):
        self.embeddings = embeddings
        self.data_df = data_df.copy()
        self.reduced_embeddings = None
        self.labels = labels
    
    def reduce_dimensionality(self, method: str = "pca", n_components: int = 2, **kwargs):
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, perplexity=30, random_state=42, verbose=True)
        else:
            raise ValueError("Неподдерживаемый метод. Используйте 'pca', 'tsne' или 'umap'.")
        
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        for i in range(n_components):
            self.data_df[f"{method}_{i+1}"] = self.reduced_embeddings[:, i]

    def cluster_data(self, method: Literal["kmeans", "dbscan", "agglomerative", "spectral"], **kwargs: Dict[str, Any]):
        if self.reduced_embeddings is None:
            raise ValueError("Запустите reduce_dimensionality() перед кластеризацией.")
        if method == "kmeans":
            model = KMeans(n_clusters=kwargs.get("n_clusters", 3), random_state=42)
        elif method == "dbscan":
            model = DBSCAN(eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5))
        elif method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=kwargs.get("n_clusters", 3))
        elif method == "spectral":
            model = SpectralClustering(n_clusters=kwargs.get("n_clusters", 3), random_state=42, assign_labels="discretize")
        else:
            raise ValueError("Неподдерживаемый метод кластеризации.")

        self.cluster_labels = model.fit_predict(self.reduced_embeddings)
        self.data_df["cluster"] = self.cluster_labels

    def compute_opacity(self, points: np.ndarray, radius: float = 0.1) -> np.ndarray:
        tree = cKDTree(points)
        densities = np.array([len(tree.query_ball_point(p, radius)) for p in points])
        
        min_density = np.min(densities)
        max_density = np.max(densities)

        if max_density == min_density:
            return np.ones_like(densities)

        opacities = 0.3 + (densities - min_density) / (max_density - min_density) * 0.7
        return np.clip(opacities, 0.3, 1.0)

    def visualize(
        self, 
        method: Literal["pca", "tsne"], 
        n_components: int = 2, 
        **kwargs
    ):
        
        use_clusters = kwargs.get("use_clusters", False)
        use_opacity = kwargs.get("use_opacity", False)
        title = kwargs.get("title",  "Embedding Visualization") 
        colorscale_name = kwargs.get("colorscale_name", "rainbow")

        filtered_df = self.data_df.copy().reset_index(drop=True)
        
        if self.labels is not None:
            unique_labels = np.unique(self.labels)
            idx2label = kwargs.get("idx2label", {idx: f"Class {idx}" for idx in unique_labels})
            unique_labels = sorted(unique_labels, key=lambda x: idx2label[x])
            idx2label = {idx: idx2label[idx] for idx in unique_labels}
        else:
            unique_labels = [0]
            idx2label = {0: "All Data"}
            
        if len(unique_labels) <= 10:
            colors = pc.qualitative.Bold
        elif len(unique_labels) <= 20:
            colors = pc.qualitative.Light24
        else:
            colors = pc.sample_colorscale(colorscale_name, [i / (len(unique_labels) - 1) for i in range(len(unique_labels))])

        color_mapping = {idx: colors[i % len(colors)] for i, idx in enumerate(unique_labels)}

        coords_filtered = self.reduced_embeddings[filtered_df.index, :n_components]
        opacities_filtered = self.compute_opacity(coords_filtered) if use_opacity else np.ones(len(filtered_df))

        fig = go.Figure()
        
        hover_columns = kwargs.get("hover_columns", None)
        if hover_columns is None:
            hover_columns = self.data_df.columns.tolist()
        else:
            missing_columns = [col for col in hover_columns if col not in self.data_df.columns]
            if missing_columns:
                print(f"Предупреждение: Колонки {missing_columns} отсутствуют в data_df. Они будут пропущены.")
            hover_columns = [col for col in hover_columns if col in self.data_df.columns]
            
        def split_long_text(text, max_length=200):
            if isinstance(text, str) and len(text) > max_length:
                return "<br>".join([text[i:i+max_length] for i in range(0, len(text), max_length)])
            return text
        
        for col in hover_columns:
            filtered_df[col] = filtered_df[col].apply(lambda x: split_long_text(x))
                
        if use_clusters and hasattr(self, "cluster_labels") and self.cluster_labels is not None:
            unique_clusters = np.unique(self.cluster_labels)

            for cluster in unique_clusters:
                mask = (pd.Series(self.cluster_labels, index=filtered_df.index) == cluster)
                subset = filtered_df[mask]
                opacity_subset = opacities_filtered[mask] if use_opacity else 1.0

                if n_components == 2:
                    fig.add_trace(go.Scatter(
                        x=subset[f"{method}_1"],
                        y=subset[f"{method}_2"],
                        mode="markers",
                        marker=dict(
                            size=8,
                            opacity=opacity_subset,
                        ),
                        name=f"Cluster {cluster}",
                        customdata=subset[hover_columns],
                        hovertemplate="<br>".join([
                            f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(hover_columns)
                        ])
                    ))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=subset[f"{method}_1"],
                        y=subset[f"{method}_2"],
                        z=subset[f"{method}_3"],
                        mode="markers",
                        marker=dict(
                            size=5,
                        ),
                        name=f"Cluster {cluster}",
                        customdata=subset[hover_columns],
                        hovertemplate="<br>".join([
                            f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(hover_columns)
                        ])
                    ))
        else:
            if n_components == 2:
                for idx in unique_labels:
                    mask = (pd.Series(self.labels, index=filtered_df.index) == idx) if self.labels is not None else pd.Series([True]*len(filtered_df))
                    subset = filtered_df[mask]
                    opacity_subset = opacities_filtered[mask] if use_opacity else 1.0

                    fig.add_trace(go.Scatter(
                        x=subset[f"{method}_1"],
                        y=subset[f"{method}_2"],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=color_mapping[idx],
                            opacity=opacity_subset,
                        ),
                        name=f"{idx2label.get(idx, f'Class {idx}')}",
                        customdata=subset[hover_columns],
                        hovertemplate="<br>".join([
                            f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(hover_columns)
                        ])
                    ))
            else:
                for idx, label in idx2label.items():
                    mask = (pd.Series(self.labels, index=filtered_df.index) == idx) if self.labels is not None else pd.Series([True]*len(filtered_df))
                    subset = filtered_df[mask]

                    fig.add_trace(go.Scatter3d(
                        x=subset[f"{method}_1"],
                        y=subset[f"{method}_2"],
                        z=subset[f"{method}_3"],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=color_mapping[idx]
                        ),
                        name=f"{idx2label.get(idx, f'Class {idx}')}",
                        customdata=subset[hover_columns],
                        hovertemplate="<br>".join([
                            f"<b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(hover_columns)
                        ])
                    ))        

        plot_width = kwargs.get("plot_width", 1200)
        plot_height = kwargs.get("plot_height", 800)

        fig.update_layout(
            title=title,
            xaxis=dict(
                title=f"{method.upper()} 1 →",
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                gridcolor="lightgray",
                gridwidth=0.5,
                zeroline=True,
                zerolinecolor="black",
                zerolinewidth=1.2
            ),
            yaxis=dict(
                title=f"{method.upper()} 2 →",
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                gridcolor="lightgray",
                gridwidth=0.5,
                zeroline=True,
                zerolinecolor="black",
                zerolinewidth=1.2
            ),
            template="plotly_white",
            width=plot_width,
            height=plot_height,
            legend_title="Class Labels"
        )

        if n_components == 3:
            fig.update_layout(scene=dict(
                xaxis=dict(title=f"{method.upper()} 1 →"),
                yaxis=dict(title=f"{method.upper()} 2 →"),
                zaxis=dict(title=f"{method.upper()} 3 →"),
            ))

        fig.show()

