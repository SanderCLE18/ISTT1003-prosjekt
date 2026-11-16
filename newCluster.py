import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from itertools import groupby
from IPython import display
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


K = 5
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

KLYNGENAVN = ["Store sentrale", "Kjempesentrale", "Små sentrale", "Dårlige", "Langt fra sentrum"]

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
ax.set_ylabel("Avstand (Ward)")
ax.set_xlabel("Grupper av boliger")
ax.set_title("Dendrogram, hierarkisk klynging")
plt.show()

# Dendrogram med grupperinger
cut = (link[-K, 2] + link[-(K - 1), 2]) / 2

fig, ax = plt.subplots(figsize=(8, 4))
dendro_k = dendrogram(link, truncate_mode="lastp", p=30, no_labels=True, ax=ax,
                      color_threshold=cut, above_threshold_color="grey")

i = 0
for nr, (farge, gruppe) in enumerate(groupby(dendro_k["leaves_color_list"]), start=1):
    n = len(list(gruppe))
    ax.add_patch(Rectangle((10 * i + 1, 0), 10 * n - 2, cut * 0.95,
                           fill=False, ec=farge, lw=2, ls="--", alpha=0.5))
    ax.text(10 * i + 5 * n, cut * 0.97, f"Klynge {nr}",
            ha="center", va="bottom", fontsize=9, color=farge)
    i += n

ax.grid(False)
ax.set_ylabel("Avstand (Ward)")
ax.set_xlabel("Grupper av boliger")
ax.set_title(f"Dendrogram med {K} klynger")
plt.show()
# Dendrogrammet gir 5 grupper/K = 5


#Lager klyngene, og finner antallet profesjonelle og medianpris i hver klynge.
kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEED)
data["klynge"] = kmeans.fit_predict(X)

klynger = data.groupby("klynge")[KLYNGEVARIABLER].mean().round(2)
klynger["antall"] = data.groupby("klynge").size()
klynger["andel prof. (%)"] = (data.groupby("klynge")["biz"].mean() * 100).round(1)
klynger["andel helg"] = (data.groupby("klynge")["helg"].mean() * 100).round(1)
klynger["medianpris"] = data.groupby("klynge")["realSum"].median().round(1)

#Klyngenes posisjon
ncol = min(K, 3)
nrad = int(np.ceil(K/ncol))

fig, axes = plt.subplots(nrows=nrad, ncols=ncol, figsize=(4*ncol, 3.6*nrad),
                         sharex=True, sharey=True, squeeze=False)

#Lager et "kart" for hver klynge
for k, ax in enumerate(axes.flatten()):
    if k>=K:
        ax.set_visible(False)
        continue
    ax.scatter(data["lng"], data["lat"], s=3, color = "#d0d0d0", linewidth=0)
    d=data[data["klynge"] == k]
    ax.scatter(d["lng"], d["lat"], s=6, color = "#78081C", linewidth=0)
    ax.set_title(f"Klynge {k + 1}, Antall = {len(d)}")
    ax.grid(False)
fig.supxlabel("Lengdegrad")
fig.supylabel("Breddegrad")
fig.tight_layout()
plt.show()

# Lager heatmap av klyngene.
profil = pd.DataFrame(X, columns=KLYNGEVARIABLER).groupby(data["klynge"]).mean()
vmax = np.abs(profil.to_numpy()).max()
fig, ax = plt.subplots(figsize=(12, 8))
bilde = ax.imshow(profil.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(KLYNGEVARIABLER)))
ax.set_xticklabels([KLYNGEVARIABLER[v] for v in range(len(KLYNGEVARIABLER))], ha = "right", rotation=45)
ax.set_yticks(range(K))
ax.set_yticklabels([f"{KLYNGENAVN[k]}" for k in range(K)])
ax.grid(False)

for i in range(K):
    for j in range(len(KLYNGEVARIABLER)):
        verdi = profil.iat[i, j]
        tekst = f"{verdi:.1f}".replace("-0.0", "0.0")
        ax.text(j, i, tekst, ha="center", va="center", color="white" if abs(verdi) > 0.5*vmax else "black")

fig.colorbar(bilde, ax=ax, label = "Standardavvik fra snittet")
ax.set_title("Klyngeprofiler")
plt.show()

#Stolpediagram av medianpris hverdag vs. helg for profesjonelle (biz) i hver klynge
biz = data[data["biz"] == 1]
median_biz = biz.groupby(["klynge", "helg"])["realSum"].median().unstack("helg")
antall_biz = biz.groupby(["klynge", "helg"]).size().unstack("helg")
okning = (median_biz[1] / median_biz[0] - 1) * 100

nbiz = data[data["biz"] == 0]
median_nbiz = nbiz.groupby(["klynge", "helg"])["realSum"].median().unstack("helg")
antall_nbiz = nbiz.groupby(["klynge", "helg"]).size().unstack("helg")
nbokning = (median_nbiz[1] / median_nbiz[0] - 1) * 100


x = np.arange(K)
bredde = 0.36
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.19, median_biz[0], bredde, color="#b8707a", label="Hverdag")
ax.bar(x + 0.19, median_biz[1], bredde, color="#78081C", label="Helg")

#Skriver prosentvis endring over hvert stolpepar prof
for k in range(K):
    ax.text(x[k], median_biz.loc[k].max() + 4, f"{okning[k]:+.1f} %", ha="center", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([f"{KLYNGENAVN[k]}\nn = {antall_biz.loc[k, 0]} / {antall_biz.loc[k, 1]}" for k in range(K)])
ax.grid(axis="x", visible=False)
ax.set_ylim(0, median_biz.to_numpy().max() * 1.18)
ax.set_ylabel("Medianpris (€)")
ax.set_title("Medianpris for profesjonelle utleiere, hverdag vs. helg")
ax.legend(frameon=False, loc="upper left", ncols=2)
plt.show()


x = np.arange(K)
bredde = 0.36
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.19, median_nbiz[0], bredde, color="#b8707a", label="Hverdag")
ax.bar(x + 0.19, median_nbiz[1], bredde, color="#78081C", label="Helg")

#Skriver prosentvis endring over hvert stolpepar uprof
for k in range(K):
    ax.text(x[k], median_nbiz.loc[k].max() + 4, f"{nbokning[k]:+.1f} %", ha="center", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([f"{KLYNGENAVN[k]}\nn = {antall_nbiz.loc[k, 0]} / {antall_nbiz.loc[k, 1]}" for k in range(K)])
ax.grid(axis="x", visible=False)
ax.set_ylim(0, median_nbiz.to_numpy().max() * 1.18)
ax.set_ylabel("Medianpris (€)")
ax.set_title("Medianpris for uprofesjonelle utleiere, hverdag vs. helg")
ax.legend(frameon=False, loc="upper left", ncols=2)
plt.show()
