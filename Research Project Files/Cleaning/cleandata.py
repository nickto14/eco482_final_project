import pandas as pd
import numpy as np

df = pd.read_csv('D:/Users/nickt/ECO482/Research Project Files/Cleaning/2026-01-MD.csv')

df['Slope'] = df['GS10'] - df['TB3MS']
df['Slope'] = df['Slope'].round(5)
df['Inversion'] = (df['Slope'] < 0).astype(int)

print(df.head())

num_vars = len(df.columns)
print(num_vars)


###############################################################
# General data cleaning steps:
###############################################################

## dropping variables with significant number of missing entries

df.drop(['ACOGNO', 'ANDENOx', 'TWEXAFEGSMTHx'], axis=1, inplace=True)

## perform linear interpolation on UMCSENTx to fill in missing values

df['UMCSENTx'] = df['UMCSENTx'].interpolate(method='linear', limit_direction='both')

## Transform codes will get deleted after removing the first few years. save them here

transform_codes = df.iloc[0]
df = df.iloc[1:]

## Dropping data before January 1963 and after January 2025 Due to many missing entries:

df['sasdate'] = pd.to_datetime(df['sasdate'])
df = df[(df['sasdate'] >= '1963-01-01') & (df['sasdate'] <= '2025-01-01')]

## Verify that there are no remaining missing values
missing_counts = df.isnull().sum()
variables_with_nas = missing_counts[missing_counts > 0]
print("Variables with empty entries:")
print(variables_with_nas.sort_values(ascending=False))

###############################################################
# Export a non-transformed version of the cleaned dataset for 
# use in the EDA and modeling scripts.
###############################################################

output_path = 'D:/Users/nickt/ECO482/Research Project Files/Cleaning/FRED_MD_Cleaned_NonTransformed.csv'
df.to_csv(output_path, index=False)

print(f"Cleaned data exported successfully to: {output_path}")

###############################################################
# Apply the transformations to the variables according to the codes in the appendix. 
# This is a crucial step, as it ensures that all variables are on the same footing for analysis and modeling. 
###############################################################

def apply_transform(x, code):
    """Applies FRED-MD transformation codes 1-7."""
    if code == 1:   # No transformation
        return x
    elif code == 2:  # First difference
        return x.diff()
    elif code == 3:  # Second difference
        return x.diff().diff()
    elif code == 4:  # Log
        return np.log(x)
    elif code == 5:  # First difference of logs
        return np.log(x).diff()
    elif code == 6:  # Second difference of logs
        return np.log(x).diff().diff()
    elif code == 7:  # Percent change
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

## 3. Handling the "Empty" rows created by the shift
## The last 6 months (Aug 2024 - Jan 2025) will have NaN for 'Target_6M', so drop them

df.dropna(subset=['Target_6M'], inplace=True)

print(df[['sasdate', 'Slope', 'Inversion', 'Target_6M']].tail(10))

###############################################################
# Export the cleaned and transformed dataset to a new CSV file
###############################################################

output_path = 'D:/Users/nickt/ECO482/Research Project Files/Cleaning/FRED_MD_Cleaned.csv'
df.to_csv(output_path, index=False)

print(f"Cleaned data exported successfully to: {output_path}")

###############################################################
# This function can be imported into the EDA and modeling scripts 
# to load the cleaned dataset and prepare the feature matrix X and target vector y.
###############################################################

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['sasdate', 'Slope', 'Inversion', 'Target_6M'])
    y = df['Target_6M']
    return X, y