import numpy as np
import sqlite3
import pandas as pd
import requests

# Getting GRBs from GRBweb
r = requests.get("https://icecube.wisc.edu/~grbweb_public/GRBweb2.sqlite")
f = open("GRBweb2.sqlite", "wb").write(r.content)
db = sqlite3.connect("GRBweb2.sqlite")
df = pd.read_sql_query("SELECT * from Summary", db)
df.GRB_name = df.GRB_name.str.replace("*", "")
df.set_index("GRB_name", inplace=True)

print("Total number of GRBs:", len(df))

# Filter those with known redshift and clean up
df = df[~np.isnan(df.redshift) & (df.redshift > 0)]
df = df[["mjd", "ra", "decl", "redshift", "fluence"]]
df = df.rename(columns={"decl": "dec", "redshift": "z"})
df["source"] = "GRBweb"
df[df.fluence < 0]["fluence"] = np.nan

# final formatting and saving
df.ra = df.ra.round(3)
df.dec = df.dec.round(3)
df.mjd = df.mjd.round(5)
df.sort_values(by="mjd", ascending=False, inplace=True)
df.to_csv("inputs/grbs/grb_catalog_grbweb.csv", index_label="name")
