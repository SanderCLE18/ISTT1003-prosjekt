import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

K = 4
SEED = 67

KLYNGEVARIABLER = [
    "person_capacity",
    "bedrooms",
    "dist",
    "metro_dist",
    "attr_index_norm",
    "rest_index_norm",
    "cleanliness_rating",
    "guest_satisfaction_overall",
]

#Basically bare konfigurering av fremtidige plots.
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#d6d6d6",
    "axes.grid": True,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


#Leser inn yo, og lager en verdi for om det er helg eller ikke
hverdag = pd.read_csv('lisbon_weekdays.csv')
hverdag["helg"] = 0

helg = pd.read_csv('lisbon_weekends.csv')
helg["helg"] = 1

#slår dataene sammen i én fil
data = pd.concat([hverdag, helg], ignore_index=True)
data.drop(columns=["Unnamed: 0"], errors="ignore")
data["biz"] = data["biz"].astype(int)

print(f"Antall annonser: {len(data)}")
print(data["room_type"].value_counts())

#Fjerner enheter som ikke er hele leiligheter/hus
data = data[data["room_type"] == "Entire home/apt"].copy()
data = data.dropna(subset=KLYNGEVARIABLER + ["realSum"]).reset_index(drop=True)

#Finner K
X = StandardScaler().fit_transform(data[KLYNGEVARIABLER])
k_verdier = list(range(1, 11))
inertia = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X).inertia_ for k in k_verdier]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k_verdier, inertia, marker="o", color="#78081C", linewidth=2, markersize=6)
ax.set_xticks(k_verdier)
ax.set_xlabel("Antall klynger (k)")
ax.set_ylabel("Sum av kvadrerte avstander\ninnad i klyngene")
ax.set_title("Albuekurve for k-means")
plt.show()

#verifiserer K ved hjelp av dendrogram.
rng = np.random.default_rng(SEED)
utvalg = rng.choice(len(data), size=min(2000, len(data)), replace=False)

#link = linkage(X[utvalg], method="complete", metric="euclidean")
#link = linkage(X[utvalg], method="single", metric="euclidean")
link = linkage(X[utvalg], method="ward", metric="euclidean")
#link = linkage(X[utvalg], method="centroid", metric="euclidean")

fig, ax = plt.subplots(figsize=(8, 4))
dendro = dendrogram(link, truncate_mode="lastp", p = 30, no_labels=True, ax=ax)

ax.grid(False)
ax.set_ylabel("Avstand (complete)")
ax.set_xlabel("Grupper av boliger")
ax.set_title("Dendrogram, hierarkisk klynging")
plt.show()
# Dendrogrammet gir 4 grupper/K = 4