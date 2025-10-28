# scripts/graph_build_example.py
# Graph Build Example – Collaboration & Leadership (robust for disconnected graphs)

import os
from itertools import combinations
import pandas as pd
import networkx as nx

DATA_DIR = os.path.join('..', 'data')
INPUT = os.path.join(DATA_DIR, 'sample_ckp_labeled.csv')
OUTPUT = os.path.join(DATA_DIR, 'graph_metrics_sample.csv')

df = pd.read_csv(INPUT)

# ---- 1) Bangun edge kolaborasi (placeholder):
# co-pegawai pada unit & tanggal yang sama. Ubah sesuai data riil: co-task, co-rapat, disposisi, dsb.
pairs = []
for (unit, tgl), grp in df.groupby(['unit', 'tanggal']):
    ids = list(grp['pegawai_id'].unique())
    for i, j in combinations(ids, 2):
        pairs.append((i, j))

G = nx.Graph()
G.add_edges_from(pairs)

# Jika graf kosong, keluarkan CSV kosong dan selesai
if G.number_of_nodes() == 0:
    pd.DataFrame(columns=['pegawai_id','degree','betweenness','eigenvector','community']).to_csv(OUTPUT, index=False)
    print("No nodes/edges; wrote empty metrics CSV.")
    raise SystemExit(0)

# ---- 2) Degree & Betweenness (aman di graf tak terhubung)
deg = dict(G.degree())
btw = nx.betweenness_centrality(G, normalized=True)

# ---- 3) Eigenvector centrality: hitung per connected component
eig = {n: 0.0 for n in G.nodes()}  # default 0
for comp_nodes in nx.connected_components(G):
    H = G.subgraph(comp_nodes).copy()
    try:
        vals = nx.eigenvector_centrality_numpy(H)  # cepat & stabil pada subgraf terhubung
    except Exception:
        # Fallback ke power method jika numpy version bermasalah
        vals = nx.eigenvector_centrality(H, max_iter=1000, tol=1e-06)
    eig.update(vals)

# ---- 4) Community (opsional): gunakan label -1 jika tidak memasang komunitas
# Untuk komunitas Louvain (butuh python-louvain):
try:
    import community as community_louvain  # pip install python-louvain
    part = community_louvain.best_partition(G)  # dict: node -> community id
    comm = part
except Exception:
    # fallback: -1 untuk semua node
    comm = {n: -1 for n in G.nodes()}

# ---- 5) Tulis hasil
rows = []
for n in G.nodes():
    rows.append({
        'pegawai_id': n,
        'degree': float(deg.get(n, 0)),
        'betweenness': float(btw.get(n, 0.0)),
        'eigenvector': float(eig.get(n, 0.0)),
        'community': int(comm.get(n, -1))
    })

pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print(f"Wrote {len(rows)} rows to {OUTPUT}")
