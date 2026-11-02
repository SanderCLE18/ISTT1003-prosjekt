import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score


def add_color(color, delta):
    c = bytes.fromhex(color.lstrip("#"))
    d = bytes.fromhex(delta.lstrip("#"))
    return "#" + "".join(f"{min(a + b, 255):02x}" for a, b in zip(c, d))

# antall klynger i den endelige analysen (velges etter steg 4)
K = 4
# helgepris / ukedagspris over denne verdien regnes som "påslag"
INCREASE = 1.02

weekday = pd.read_csv('lisbon_weekdays.csv', index_col=0)
weekend = pd.read_csv('lisbon_weekends.csv', index_col=0)

KEY = ['lat', 'lng', 'room_type', 'bedrooms', 'person_capacity',
                  'host_is_superhost', 'multi', 'biz', 'cleanliness_rating']

# separerer ulike rom i samme bygg. F.eks. om et bygg har flere rom.
for d in (weekday, weekend):
    d['n'] = d.groupby(KEY).cumcount()

# Fjerner enheter som ikke finnes i annet datasett.
df = weekday.merge(weekend[KEY + ['n', 'realSum']],
                   on=KEY + ['n'], suffixes=('', '_helg'))
df = df.rename(columns={'realSum': 'pris_ukedag', 'realSum_helg': 'pris_helg'})


# Forholdstall mellom helgepris og ukedagspris for samme bolig.
df['pris_forhold'] = df['pris_helg'] / df['pris_ukedag']
df['paaslag'] = (df['pris_forhold'] > INCREASE).astype(int)
df['profesjonell'] = df['biz']

print(f'Koblede boliger: {len(df)} (ukedag: {len(weekday)}, helg: {len(weekend)})')
print(f'Andel med helgepåslag: {df["paaslag"].mean():.1%}')
print(f'Andel profesjonelle:   {df["profesjonell"].mean():.1%}')
print(f'Andel av alle boiger som både er profesjonelle og har påslag {(df['profesjonell'] & df['paaslag']).mean():.1%}')
print(f'Andel profesjonelle med helgepåslag: {df[df["profesjonell"] == 1]["paaslag"].mean():.1%}\n')

# Etter kjøring kan vi se at det ikke spiller rolle om de er profesjonelle eller ikke.
# Men prøver fortsatt å finne bestemmende faktor


X = pd.DataFrame(index=df.index)


# room_type er kategorisk, så den gjøres om til 0/1-kolonner (one-hot).
X = X.join(pd.get_dummies(df['room_type'], prefix='type', dtype=float))
# Få boliger har 4-10 soverom; kapper ved 4 så de ikke drar i klyngene.
X['bedrooms'] = df['bedrooms'].clip(upper=4)
X['person_capacity'] = df['person_capacity']

# Avstandene og indeksene er sterkt høyreskjeve; log-transformasjon gjør
# fordelingene mer symmetriske slik at få ekstreme verdier ikke dominerer.
X['log_dist'] = np.log(df['dist'])
X['log_metro_dist'] = np.log(df['metro_dist'])
X['log_attr_index'] = np.log(df['attr_index_norm'])
X['log_rest_index'] = np.log(df['rest_index_norm'])

# Standardisering (snitt 0, std 1) slik at alle variabler teller likt.
# K-means bruker avstand, og uten dette ville variabler med stor skala vinne.
X_scaled = StandardScaler().fit_transform(X)


#Antallet klynger
ks = range(2, 11)
# variasjon innen klynge
treghet = []
# silhouette-score: høyere = bedre adskilte og kompakte klynger
silhuett = []

for k in ks:
    modell = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++').fit(X_scaled)
    treghet.append(modell.inertia_)
    silhuett.append(silhouette_score(X_scaled, modell.labels_))
    print(f'k={k}: inertia={treghet[-1]:.0f}, silhouette={silhuett[-1]:.3f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(ks, treghet, marker='o')
ax1.set_xlabel('Antall klynger (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Albuemetoden (let etter "knekk")')
ax2.plot(ks, silhuett, marker='o')
ax2.set_xlabel('Antall klynger (k)')
ax2.set_ylabel('Silhouette-score')
ax2.set_title('Silhouette (høyere er bedre)')
plt.tight_layout()
plt.show()


# Klyngeanalyse
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['klynge'] = kmeans.fit_predict(X_scaled)


profil = df.groupby('klynge').agg(
    antall=('klynge', 'size'),
    andel_hele_boliger=('room_type', lambda s: (s == 'Entire home/apt').mean()),
    andel_privat_rom=('room_type', lambda s: (s == 'Private room').mean()),
    soverom=('bedrooms', 'mean'),
    kapasitet=('person_capacity', 'mean'),
    avstand_sentrum=('dist', 'mean'),
    avstand_metro=('metro_dist', 'mean'),
    attraksjoner=('attr_index_norm', 'mean'),
    restauranter=('rest_index_norm', 'mean'),
    andel_biz=('biz', 'mean'),
    andel_paaslag=('paaslag', 'mean'),
)
print(profil.round(2).to_string(), '\n')

trekk = df[['person_capacity', 'bedrooms', 'dist', 'metro_dist', 'attr_index_norm',
            'rest_index_norm', 'cleanliness_rating', 'guest_satisfaction_overall',
            'host_is_superhost', 'multi', 'biz', 'pris_ukedag']].astype(float)
trekk['hele_boliger'] = (df['room_type'] == 'Entire home/apt').astype(float)

skog = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=42)
print('AUC:', cross_val_score(skog, trekk, df['paaslag'], cv=5, scoring='roc_auc').mean().round(2))
skog.fit(trekk, df['paaslag'])
v = permutation_importance(skog, trekk, df['paaslag'], n_repeats=10, random_state=42, scoring='roc_auc')
print(pd.Series(v.importances_mean, index=trekk.columns).sort_values(ascending=False).round(3).to_string())


df['prisokning'] = (df['pris_forhold'] - 1) * 100
FARGER = {0: '#2a78d6', 1: '#eb6834'}
NAVN = {0: 'Ikke-profesjonell', 1: 'Profesjonell'}
typer = ['Entire home/apt', 'Private room', 'Shared room']

fig, ax = plt.subplots(figsize=(7,5))
pos = np.arange(len(typer))

#boksplot, ser på prisøkning for de forskjellige enhetskategoriene,
for k,biz in enumerate((0,1)):
    data = [df[(df['room_type'] == t) & (df['profesjonell'] == biz)]['prisokning'] for t in typer]
    bp = ax.boxplot(data, positions = pos+(k-0.5)*0.35, widths = 0.3,
                    patch_artist = True, showfliers=False, medianprops={'color' : 'black'})
    i = 0
    for boks in bp['boxes']:
        boks.set(facecolor=add_color(FARGER[biz], str(i)+"f0000"), alpha=0.67)
        ax.plot([], [], 's', color = FARGER[biz], label = NAVN[biz])
        i += 1
ax.axhline(0, color = 'gray', linewidth = 1)
ax.set_xticks(pos, typer)
ax.set_ylabel('Prisøkning helg vs. ukedag (%)')
ax.legend()
plt.show()

#Scatterplot for relasjon mellom prisøkning og distanse til sentrum.
for biz in (0,1):
    d = df[df['profesjonell'] == biz]
    plt.scatter(d['dist'], d['prisokning'].clip(-30, 60), s=12, alpha=0.5,
                color=FARGER[biz], label=NAVN[biz], linewidths=0)
plt.xlabel('Distance (km)')
plt.ylabel('Prisokning (%)')
plt.legend()
plt.show()

# Kart over Lisboa der hver prikk er en bolig farget etter klynge.
"""plt.figure(figsize=(7, 6))
plt.scatter(df['lng'], df['lat'], c=df['klynge'], cmap='viridis',
            edgecolors='k', linewidths=0.3, s=20)
plt.colorbar(label='Klynge')
plt.xlabel('Lengdegrad')
plt.ylabel('Breddegrad')
plt.title(f'Boliger i Lisboa fordelt på {K} klynger')
plt.show()"""

"""
plt.figure(figsize=(7, 6))
sc = plt.scatter(df['lng'], df['lat'], c=df['pris_forhold'].clip(0.9, 1.3),
                 cmap='coolwarm', s=20)
plt.colorbar(sc, label='Helgepris / ukedagspris')
plt.xlabel('Lengdegrad')
plt.ylabel('Breddegrad')
plt.show()
"""