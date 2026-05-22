# KMeans Baseline Notes

## Best configuration

- Best feature set: `geometry_only`
- Best k: `3`
- Silhouette score: `0.2430`
- Davies-Bouldin index: `1.6064`
- Calinski-Harabasz score: `2665.3316`

## Cluster size

- Cluster 0: 3440 samples
- Cluster 1: 4470 samples
- Cluster 2: 2243 samples

## Energy interpretation

- Most stable cluster by mean relative energy: `0`
- Least stable cluster by mean relative energy: `1`

## Suggested interpretation

- Cluster 0 (dense-stable-high-coordination): 3440 samples, mean relative energy = 243.72 meV, mean density = 3.481, mean volume/atom = 5.737.
- Cluster 1 (dense-metastable-high-coordination): 4470 samples, mean relative energy = 380.06 meV, mean density = 3.125, mean volume/atom = 6.427.
- Cluster 2 (open-stable-low-coordination): 2243 samples, mean relative energy = 256.01 meV, mean density = 2.765, mean volume/atom = 7.218.
