###############################################################
# Read Me: 
#
# This file is for the cleaning and transformation of the FRED-MD dataset.
#
###############################################################

import pandas as pd
import numpy as np

###############################################################
# Read in Data:
###############################################################

df = pd.read_csv('D:/Users/nickt/ECO482/Research Project Files/Data/2026-01-MD.csv')

## Create Slope Variable: Slope_V = 10Y - 3M 

df['Slope'] = df['GS10'] - df['TB3MS']
df['Slope'] = df['Slope'].round(5)
df['Inversion'] = (df['Slope'] < 0).astype(int)

###############################################################
# Dropping values and Forward Filling Missing Values:
###############################################################

## Drop 10Y and 3M treasury yields, as keeping them will cause leakage.
df.drop(['GS10', 'TB3MS', 'T10YFFM', 'TB3SMFFM'], axis=1, inplace=True)

## dropping variable 'ACOGNO' as it has a significant number of missing entries
df.drop(['ACOGNO'], axis=1, inplace=True)

## Forward fill values for the variable UMCSENTx, as it seems to be recorded quartlerly until 1977, when it goes to monthly.
df['UMCSENTx'] = df['UMCSENTx'].ffill()

## Forward fill values for the variable CP3Mx and COMPAPFFx,, as they have missing values in April 2020.
df['CP3Mx'] = df['CP3Mx'].ffill()
df['COMPAPFFx'] = df['COMPAPFFx'].ffill()

###############################################################
# Transformations
###############################################################

## Transform codes will get deleted after removing the first few years. save them here:
transform_codes = df.iloc[0]
df = df.iloc[1:]

## Dropping data before January 1972 and after January 2025 due to many missing entries:
df['sasdate'] = pd.to_datetime(df['sasdate'])
df = df[(df['sasdate'] >= '1973-01-01') & (df['sasdate'] <= '2025-01-01')]

# Export a non-transformed version of the cleaned dataset for use in EDA and modeling scripts.
output_path = 'D:/Users/nickt/ECO482/Research Project Files/Data'
df.to_csv(f"{output_path}/FRED_MD_Cleaned_NonTransformed.csv", index=False)

def apply_transform(x, code):
    """Applies FRED-MD transformation codes 1-7. These codes can be found in the appendix of the FRED-MD dataset documentation."""
    if code == 1: # No transformation
        return x
    elif code == 2: # First difference
        return x.diff()
    elif code == 3: # Second difference
        return x.diff().diff()
    elif code == 4: # Log
        return np.log(x)
    elif code == 5: # First difference of logs
        return np.log(x).diff()
    elif code == 6: # Second difference of logs
        return np.log(x).diff().diff()
    elif code == 7: # Percent change
        return x.pct_change()
    return x

for col in df.columns:
    if col in transform_codes:
        code = transform_codes[col]
        df[col] = apply_transform(df[col], code)

## Transformations create NaNs in the first 1-2 rows, so drop the empty rows.
df.dropna(inplace=True)

###############################################################
# Create a 6 month forward shift of the 'Inversion' variable to use as the 
# target variable for prediction.
###############################################################

## shift(-6) moves values from the future into the current row
df['Target_6M'] = df['Inversion'].shift(-6)

## The last 6 months (Aug 2024 - Jan 2025) will have NaN for 'Target_6M', so drop them
df.dropna(subset=['Target_6M'], inplace = True)

###############################################################
# Export the cleaned and transformed dataset to a new CSV file
###############################################################

df.to_csv(f"{output_path}/FRED_MD_Cleaned_Transformed.csv", index=False)