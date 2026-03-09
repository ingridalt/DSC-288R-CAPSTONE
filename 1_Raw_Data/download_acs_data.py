#!/usr/bin/env python


# # Obtaining Raw Data 
# This project uses the American Community Survey (ACS) PUMS 1-Year national person-level microdata
# i.e. Census Data
# 2018–2023 (excluding 2020 due to the unusual circumstances of the pandemic)
# 
# Due to size constraints, the raw datasets are not tracked in GitHub
# All data can be fully reproduced by running the script, more detail on reproducibility in README file
# 
# Raw data source:
# https://www.census.gov/programs-surveys/acs/microdata.html
# https://www2.census.gov/programs-surveys/acs/data/pums/

# # CA Data 



import os
import requests
import zipfile
import io
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR  = Path(__file__).resolve().parent
BASE_DIR    = str(SCRIPT_DIR / "data_persons_ca_1yr")
YEARS       = [2018, 2019, 2021, 2022, 2023, 2024]
FILENAME    = "psam_p06.csv"
OUTPUT_PATH = os.path.join(BASE_DIR, "persons_master.csv")

def download_acs_1year_person_data(state_abbr="ca", years=YEARS):
    """
    Downloads 1-Year ACS PUMS person files. 
    """
    for year in years:
        url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_p{state_abbr}.zip"
        dest_folder = str(SCRIPT_DIR / f"data_persons_{state_abbr}_1yr" / str(year))

        if os.path.exists(dest_folder):
            response = input(
                f"Folder '{dest_folder}' already exists. Redownload? (y/n): "
            ).strip().lower()
            if response != "y":
                print(f"  Skipping {year}.")
                continue
            print(f"  Redownloading {year}...")

        os.makedirs(dest_folder, exist_ok=True)
        print(f"Downloading {year} 1-Year data from:\n  {url}")
        
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(dest_folder)
                print(f"Done: {year}")
        except Exception as e:
            print(f"Skipping {year}: {e}")

def build_master_common_columns(base_dir=BASE_DIR, years=YEARS, filename=FILENAME, output_path=OUTPUT_PATH):
    # Collect file paths  & compute  intersection of column names across years
    paths = {}
    common_cols = None

    for y in years:
        path = os.path.join(base_dir, str(y), filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for {y}: {path}")
        paths[y] = path

        # Read only header to get columns (fast)
        cols = pd.read_csv(path, nrows=0).columns
        cols_set = set(cols)

        if common_cols is None:
            common_cols = cols_set
        else:
            common_cols = common_cols.intersection(cols_set)

    #  add "year"
    common_cols = sorted(common_cols)
    if "year" in common_cols:
        # unlikely, but just in case
        common_cols.remove("year")

    print(f"Common columns across {len(years)} years: {len(common_cols)}")

  
    dfs = []
    for y in years:
        df = pd.read_csv(
            paths[y],
            usecols=common_cols,     
            low_memory=False
        )
        df["year"] = y
        dfs.append(df)
        print(f"Loaded {y}: {df.shape[0]:,} rows, {df.shape[1]} cols (incl year)")

    master = pd.concat(dfs, ignore_index=True)
    print(f"Master shape: {master.shape[0]:,} rows, {master.shape[1]} cols")
    
    before = len(master)
    master = master[master["AGEP"] > 18]
    print(f"Removed (age ≤ 18): {before - len(master):,}")
    print(f"Final rows  (19+) : {len(master):,}")
    
    master.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")

    return master, common_cols



if __name__ == "__main__":
    print(f"Starting download for years: {YEARS}")
    print(f"Output will be saved to:     {OUTPUT_PATH}\n")

    download_acs_1year_person_data()
    master_df, common_cols = build_master_common_columns()

    print("\n Data successfully downloaded! Summary:")
    print(f"  Rows    : {master_df.shape[0]:,}")
    print(f"  Columns : {master_df.shape[1]}")



