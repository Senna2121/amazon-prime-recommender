# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:09:26 2026

@author: janma
"""

import pandas as pd
import csv

# Het originele bestand openen en de gegevens opschonen
schone_rijen = []

with open("amazon_prime_titles(in).csv", 
          encoding="utf-8-sig") as bestand:

    for nummer, regel in enumerate(bestand, 1):

        regel = regel.strip().rstrip(";")

        if nummer == 1:
            regel = regel.rstrip(";")
        elif regel.startswith('"') and regel.endswith('"'):
            regel = regel[1:-1]

        regel = regel.replace('""', '"')

        try:
            rij = next(csv.reader([regel]))

            if len(rij) == 12:
                schone_rijen.append(rij)

        except csv.Error:
            continue

# De opgeschoonde gegevens omzetten naar een nieuwe dataset en opslaan
kolommen = schone_rijen[0]
df = pd.DataFrame(schone_rijen[1:], columns=kolommen)

df.to_csv("amazon_prime_clean.csv", index=False)

# Controleren of de dataset goed is ingelezen en hoeveel gegevens erin staan
print(df.head())
print(df.shape)

# De dataset verder bekijken om een beter beeld te krijgen van de gegevens

# Bekijken welke kolommen er allemaal in de dataset staan
print(df.columns.tolist())

# Controleren hoeveel gegevens er per kolom ontbreken
print(df.isnull().sum())

# Controleren of er dubbele rijen in de dataset staan
print(df.duplicated().sum())

# Bekijken hoeveel films en series er in de dataset staan
print(df["type"].value_counts())

# Bekijken welke genres het meest voorkomen in de dataset
print(df["listed_in"].value_counts().head(10))


# De bibliotheken importeren die nodig zijn om films met elkaar te vergelijken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# De genres en beschrijvingen van films samenvoegen om ze te kunnen vergelijken
df["features"] = (
    df["listed_in"].fillna("") + " " +
    df["description"].fillna("")
)

# Een eerste indruk krijgen van de samengevoegde filmgegevens
print(df[["title", "features"]].head())

# Een eerste experiment uitvoeren met cosine similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# De genres en beschrijvingen omzetten naar getallen zodat Python ermee kan rekenen
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["features"])

# Berekenen hoe sterk de films inhoudelijk op elkaar lijken
cosine_sim = cosine_similarity(tfidf_matrix)

# De eerste resultaten bekijken om te zien of de vergelijking werkt
print(cosine_sim[:5, :5])