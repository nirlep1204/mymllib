import numpy as np
import pandas as pd

class Hierarchical:
    """Agglomerative hierarchical clustering."""
    
    def __init__(self, k=3, linkage='single'):
        self.k = k
        self.linkage = linkage
        self.labels = None
        self.merge_history = []

    def fit(self, X):
        """Fit hierarchical clustering to data."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.array(X)
            
        n_samples = X_arr.shape[0]
        
        # Oversized matrix to hold distances as we add new merged clusters
        dist_matrix = np.full((n_samples * 2, n_samples * 2), np.inf)
        
        # Compute initial pairwise distances
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.sqrt(np.sum((X_arr[i] - X_arr[j])**2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                
        active_clusters = list(range(n_samples))
        cluster_sizes = {i: 1 for i in range(n_samples)}
        
        next_id = n_samples
        
        # Merge until we reach k clusters
        while len(active_clusters) > self.k:
            # Extract submatrix for currently active clusters
            idx_grid = np.ix_(active_clusters, active_clusters)
            sub_dist = dist_matrix[idx_grid]
            
            # Find the closest pair
            min_flat = np.argmin(sub_dist)
            i_idx = min_flat // len(active_clusters)
            j_idx = min_flat % len(active_clusters)
            
            c_i = active_clusters[i_idx]
            c_j = active_clusters[j_idx]
            
            min_dist = sub_dist[i_idx, j_idx]
            
            # Record merge
            new_size = cluster_sizes[c_i] + cluster_sizes[c_j]
            cluster_sizes[next_id] = new_size
            self.merge_history.append((c_i, c_j, min_dist, new_size))
            
            # Update distances for the new merged cluster
            for c in active_clusters:
                if c != c_i and c != c_j:
                    d_i = dist_matrix[c_i, c]
                    d_j = dist_matrix[c_j, c]
                    
                    if self.linkage == 'single':
                        d_new = min(d_i, d_j)
                    elif self.linkage == 'complete':
                        d_new = max(d_i, d_j)
                    elif self.linkage == 'average':
                        d_new = (cluster_sizes[c_i] * d_i + cluster_sizes[c_j] * d_j) / new_size
                    else:
                        d_new = min(d_i, d_j)
                        
                    dist_matrix[next_id, c] = d_new
                    dist_matrix[c, next_id] = d_new
                    
            # Remove old clusters, add the new one
            active_clusters.remove(c_i)
            active_clusters.remove(c_j)
            active_clusters.append(next_id)
            
            next_id += 1
            
        # Reconstruct final labels
        self.labels = np.zeros(n_samples, dtype=int)
        
        def assign_labels(cluster_id, label):
            if cluster_id < n_samples:
                self.labels[cluster_id] = label
            else:
                merge_record = self.merge_history[cluster_id - n_samples]
                assign_labels(merge_record[0], label)
                assign_labels(merge_record[1], label)
                
        for final_label, c in enumerate(active_clusters):
            assign_labels(c, final_label)
            
        return self

    def dendrogram_data(self):
        """Return merge history for dendrogram plotting."""
        return self.merge_history
