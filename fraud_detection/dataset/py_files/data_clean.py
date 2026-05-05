#loading data
import pandas as pd

df = pd.read_csv("dataset\synthetic_fraud_dataset.csv")

print(df.shape)
print(df.columns)
print(df.head())

#check missing values, duplicates
print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()

#fixing data types
print("Before conversion:\n", df.dtypes)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

bool_cols = ['IP_Address_Flag', 'Previous_Fraudulent_Activity', 'Is_Weekend']
df[bool_cols] = df[bool_cols].astype(bool)

cat_cols = ['Transaction_Type', 'Device_Type', 'Location', 
            'Merchant_Category', 'Card_Type', 'Authentication_Method']
df[cat_cols] = df[cat_cols].astype('category')

print("\nAfter conversion:\n", df.dtypes)

#feature engineering
#extracting time features
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day

#creating high risk flags
df['High_Amount'] = df['Transaction_Amount'] > 500
df['High_Distance'] = df['Transaction_Distance'] > 1000
df['High_Frequency'] = df['Daily_Transaction_Count'] > 10

#behaviour risk score
df['Behavior_Risk'] = (
    df['High_Amount'].astype(int) +
    df['High_Distance'].astype(int) +
    df['High_Frequency'].astype(int)
)

#EDA
#fraud distribution
print(df['Fraud_Label'].value_counts())

#avg amount- fraud vs normal
print(df.groupby('Fraud_Label')['Transaction_Amount'].mean())

#risk score analysis
print(df.groupby('Fraud_Label')['Risk_Score'].mean())

df.to_csv("dataset\cleaned_fraud_data.csv", index=False)