from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Hvordan henger antall soverom og avstand til sentrum sammen med Airbnb-prisen i Lisboa, når vi kontrollerer for boligtype, kapasitet og gjestetilfredshet?

DATA_PATH = Path("dataset/lisbon_weekdays.csv")
OUTPUT_DIR = Path("resultater_regresjon")
OUTPUT_DIR.mkdir(exist_ok=True)

REQUIRED_COLUMNS = [
    "realSum",
    "bedrooms",
    "dist",
    "room_type",
    "person_capacity",
    "guest_satisfaction_overall",
]


def save_figure(filename: str) -> None:
    """Save the current figure with consistent formatting."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


df = pd.read_csv(DATA_PATH)
missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
if missing_columns:
    raise ValueError(f"Datasettet mangler nødvendige kolonner: {missing_columns}")

initial_n = len(df)
df = df[REQUIRED_COLUMNS].dropna().copy()
df = df[df["realSum"] > 0].copy()
if df.empty:
    raise ValueError("Ingen gyldige observasjoner etter datarensing.")

df["ln_price"] = np.log(df["realSum"])

print(f"Observasjoner i datasettet: {initial_n}")
print(f"Observasjoner i analysen: {len(df)}")
print(f"Fjernede observasjoner: {initial_n - len(df)}")
print("\nAntall boliger etter boligtype:")
print(df["room_type"].value_counts().to_string())

# Deskriptiv statistikk til rapportens tabell.
numeric_columns = [
    "realSum",
    "bedrooms",
    "dist",
    "person_capacity",
    "guest_satisfaction_overall",
]
descriptive = df[numeric_columns].describe().T
descriptive = descriptive.rename(
    columns={
        "count": "N",
        "mean": "Gjennomsnitt",
        "std": "Standardavvik",
        "min": "Minimum",
        "50%": "Median",
        "max": "Maksimum",
    }
)
descriptive[
    ["N", "Gjennomsnitt", "Standardavvik", "Minimum", "Median", "Maksimum"]
].to_csv(OUTPUT_DIR / "deskriptiv_statistikk.csv", float_format="%.4f")

# Figur 1: prisfordeling før og etter log-transformasjon.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df["realSum"], bins=40, edgecolor="white")
axes[0].set_title("Fordeling av Airbnb-pris")
axes[0].set_xlabel("Pris (realSum)")
axes[0].set_ylabel("Antall boliger")
axes[1].hist(df["ln_price"], bins=40, edgecolor="white")
axes[1].set_title("Fordeling av log-pris")
axes[1].set_xlabel("log(pris)")
axes[1].set_ylabel("Antall boliger")
save_figure("01_prisfordeling.png")

# Figur 2: sammenhenger med de to viktigste forklaringsvariablene.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(df["bedrooms"], df["ln_price"], alpha=0.25, s=12)
bedrooms_line = np.polyfit(df["bedrooms"], df["ln_price"], 1)
bedrooms_x = np.linspace(df["bedrooms"].min(), df["bedrooms"].max(), 100)
axes[0].plot(
    bedrooms_x,
    np.polyval(bedrooms_line, bedrooms_x),
    color="red",
    linewidth=2,
)
axes[0].set_title("Soverom og log-pris")
axes[0].set_xlabel("Antall soverom")
axes[0].set_ylabel("log(pris)")
axes[1].scatter(df["dist"], df["ln_price"], alpha=0.25, s=12)
dist_line = np.polyfit(df["dist"], df["ln_price"], 1)
dist_x = np.linspace(df["dist"].min(), df["dist"].max(), 100)
axes[1].plot(
    dist_x,
    np.polyval(dist_line, dist_x),
    color="red",
    linewidth=2,
)
axes[1].set_title("Avstand og log-pris")
axes[1].set_xlabel("Avstand til sentrum")
axes[1].set_ylabel("log(pris)")
save_figure("02_sammenhenger_med_regresjonslinje.png")

# Figur 3: pris etter boligtype.
df.boxplot(column="realSum", by="room_type", ax=plt.gca(), grid=False)
plt.suptitle("")
plt.title("Pris fordelt på boligtype")
plt.xlabel("Boligtype")
plt.ylabel("Pris (realSum)")
save_figure("03_pris_etter_boligtype.png")

# Multippel lineær regresjon. Entire home/apt er referansekategori.
formula = (
    "ln_price ~ bedrooms + dist + C(room_type) + "
    "person_capacity + guest_satisfaction_overall"
)
model = smf.ols(formula, data=df).fit()

print("\nRegresjonsmodell:")
print(model.summary())
(OUTPUT_DIR / "regresjonssammendrag.txt").write_text(
    model.summary().as_text(), encoding="utf-8"
)

variable_descriptions = pd.DataFrame(
    [
        ("realSum", "Numerisk", "Responsvariabel", "Airbnb-pris"),
        ("bedrooms", "Numerisk", "Forklaringsvariabel", "Antall soverom"),
        ("dist", "Numerisk", "Forklaringsvariabel", "Avstand til sentrum"),
        ("room_type", "Kategorisk", "Kontrollvariabel", "Boligtype"),
        (
            "person_capacity",
            "Numerisk",
            "Kontrollvariabel",
            "Antall personer boligen har kapasitet til",
        ),
        (
            "guest_satisfaction_overall",
            "Numerisk",
            "Kontrollvariabel",
            "Gjestevurdering",
        ),
    ],
    columns=["Variabel", "Type", "Rolle", "Beskrivelse"],
)
variable_descriptions.to_csv(OUTPUT_DIR / "variabeloversikt.csv", index=False)

# Regresjonstabell med standardfeil, konfidensintervall og prosenttolkning.
conf_int = model.conf_int()
regression_table = pd.DataFrame(
    {
        "Variabel": model.params.index,
        "Koeffisient": model.params.values,
        "Standardfeil": model.bse.values,
        "t-verdi": model.tvalues.values,
        "p-verdi": model.pvalues.values,
        "KI 2.5%": conf_int[0].values,
        "KI 97.5%": conf_int[1].values,
        "Prosentvis endring": 100 * np.expm1(model.params.values),
        "Prosent KI 2.5%": 100 * np.expm1(conf_int[0].values),
        "Prosent KI 97.5%": 100 * np.expm1(conf_int[1].values),
    }
)
regression_table.to_csv(
    OUTPUT_DIR / "regresjonstabell.csv", index=False, float_format="%.6f"
)

print("\nTolkning av hovedvariablene:")
for variable in ("bedrooms", "dist"):
    coefficient = model.params[variable]
    lower, upper = model.conf_int().loc[variable]
    print(
        f"{variable}: {100 * np.expm1(coefficient):.2f}% "
        f"(95% KI: {100 * np.expm1(lower):.2f}% til "
        f"{100 * np.expm1(upper):.2f}%)"
    )

# Modelltilpasning og tester som kan omtales i metode- og resultatdelen.
diagnostics = pd.DataFrame(
    {
        "mål": [
            "Antall observasjoner",
            "R-squared",
            "Justert R-squared",
            "RMSE på log-skala",
            "Breusch-Pagan p-verdi",
            "Jarque-Bera p-verdi",
            "Skjevhet i residualer",
            "Kurtose i residualer",
        ],
        "verdi": [
            len(df),
            model.rsquared,
            model.rsquared_adj,
            np.sqrt(np.mean(model.resid**2)),
            het_breuschpagan(model.resid, model.model.exog)[1],
            stats.jarque_bera(model.resid).pvalue,
            stats.skew(model.resid),
            stats.kurtosis(model.resid, fisher=False),
        ],
    }
)
diagnostics.to_csv(OUTPUT_DIR / "modelldiagnostikk.csv", index=False)
print("\nModelltilpasning:")
print(diagnostics.to_string(index=False))

# Variance Inflation Factor for numeriske forklaringsvariabler.
vif_data = df[
    ["bedrooms", "dist", "person_capacity", "guest_satisfaction_overall"]
].copy()
vif_design = sm.add_constant(vif_data)
vif_table = pd.DataFrame(
    {
        "Variabel": vif_design.columns,
        "VIF": [
            variance_inflation_factor(vif_design.values, i)
            for i in range(vif_design.shape[1])
        ],
    }
)
vif_table.to_csv(OUTPUT_DIR / "vif.csv", index=False, float_format="%.4f")
print("\nVIF:")
print(vif_table.to_string(index=False))

# Figur 4: residualanalyse.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(model.fittedvalues, model.resid, alpha=0.25, s=12)
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_title("Residualer mot predikerte verdier")
axes[0].set_xlabel("Predikert log-pris")
axes[0].set_ylabel("Residual")
sm.qqplot(model.resid, line="45", ax=axes[1], fit=True)
axes[1].set_title("Q-Q-plott for residualer")
save_figure("04_residualanalyse.png")

print(f"\nAlle tabeller og figurer er lagret i: {OUTPUT_DIR}/")