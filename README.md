
## **California Poverty Risk Prediction**

Using ACS PUMS Census Data to Identify At-Risk Individuals Before They Fall Below the Poverty Line

---

#### **Why This Matters:** 
Poverty is one of the most consequential and measurable public health challenges in California. California has one of the highest poverty rates in the United States when adjusted for cost of living. With the complexity of risk detection, ACS provided vast features and scale of data makes this an ideal problem for Machine Learning. 
<br>**Our goal:** Build a scalable pre-screening framework that state agencies and social service organizations can use to proactively identify individuals at risk, enabling early intervention, equitable resource distribution, and prevention before families slide further into poverty. </br>


#### Data
Dataset: https://www2.census.gov/programs-surveys/acs/data/pums/ 
<br>Source: U.S. Census Bureau — American Community Survey (ACS) PUMS 1-Year Estimates </br>
Geography: California adults (age 19+)
Years: 2018, 2019, 2021, 2022, 2023 (train) → 2024 (test)
Raw size: ~2.3M records, 275 columns
Split strategy: Temporal — train on historical years, test on 2024 holdout to simulate real-world deployment

#### Env note: 
All FT-Transformer training was conducted on an NVIDIA RTX A5000 GPU (24 GB VRAM, CUDA 12.2) within a 16 GB RAM environment; a GPU is required to train the FT-Transformer in a reasonable time, as the quadratic complexity of multi-head self-attention with respect to sequence length makes CPU training prohibitively slow at this data scale (~710K training samples).

All other models used CPU 

#### Modeling Approach
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Role</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
<tr>
      <td>Multinomial Logistic Regression</td>
      <td>Baseline</td>
      <td>OHE encoding, L2 regularization, StandardScaler on numerics</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Advanced</td>
      <td>Handles non-linearity, built-in feature importance</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>Advanced</td>
      <td>Primary model, elbow analysis used to select optimal features</td>
    </tr>
</tbody>
</table>

#### Project Structure 
<pre>
├── 1_Raw_Data/
│   └── data_persons_ca_1yr/persons_master.csv
|   └── download_acs_data.py Loading of raw data and conversion into csv format from ACS FTP Site
├── 2_EDA/
│   └── EDA notebooks and visualizations through EDA_Master.ipynb
├── 3_Data_Preprocessing/
│   ├── Master_Preprocessing.ipynb  # Master Preprocessing notebook generating train_engineered.csv and test_engineered.csv used globally across all files
│   ├── preprocessing_data/
│       ├── train_engineered.csv
│       └── test_engineered.csv
    └──  FT-Preprocessing.ipynb # Required preprocessing for the FT Dataset (converts files to parquet, this was done due to env limitations in Datahub)
├── 4_Models/
│   ├── 4a_Baseline.ipynb 
│   ├── 4b_RandomForest_Model.ipynb
│   └── 4c_XGBoost_Model.ipynb
│ 
├── 5_Feature_Transformer/
│   ├── FT-Model-Build.ipynb # FT model build <u>Note:</u> To run this notebook, GPU is required, please see note Env Note above
│   ├── data_to_tesnor.py #  python class that  converts data to tensor format 
│   └── ft_model_def.py # python class that defines the FT Model layers 
│
├── README.md
├── Makefile
├── Requirements.txt
</pre>


#### **Conda Environment Setup**
<u>To use the Conda env, user must run the following commands </u>

`conda init`

`make setup`

*Note:* conda init modifies your shell configuration. 
After running it, you will need to open a new terminal (or run source ~/.bashrc / source ~/.zshrc)
for the changes to take effect before proceeding with make setup.

<u>After setup, activate the environment in your terminal:</u>
`conda activate poverty_prediction`

Alternatively, if working in Jupyter, simply select the "Python (Poverty Project)" kernel from the kernel menu — no terminal activation needed.

<u>Run this command to update any new dependencies from other users</u>

`make update `

<u>To update requirements.txt after installing new ones run </u>

`make freeze`

#### Raw Data Download

Due to file size constraints, raw data is **not tracked in GitHub** and must be downloaded locally by running the provided script.

This project uses the [American Community Survey (ACS) PUMS 1-Year](https://www.census.gov/programs-surveys/acs/microdata.html) 
person-level microdata for California, covering **2018, 2019, 2021, 2022, 2023, and 2024** 
(2020 is excluded due to pandemic-related data collection disruptions).

** Note: This script must be run from the root of the project. To download the raw data run:**
```
python 1_Raw_Data/download_acs_data.py
```

The script will:
1. Download each year's data directly from the [Census Bureau FTP site](https://www2.census.gov/programs-surveys/acs/data/pums/)
2. Prompt you before overwriting any year that has already been downloaded
3. Combine all years into a single file, keeping only columns common across all years
4. Filter to adults (age 19+)
5. Save the final dataset to `1_Raw_Data/data_persons_ca_1yr/persons_master.csv`

> **Note:** The full download is large. Expect the process to take several minutes depending on your connection speed.

