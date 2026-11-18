import pandas as pd

def load(name):
    df =   pd.read_csv(name)
    return df
    
    
def clean(df):
    df["name"] = df["name"].str.strip()
    df["city"] = df["city"].str.strip()
    
    df["city"] = df["city"].str.title()
    
    df["age"] = pd.to_numeric(df["age"], errors = 'coerce')
    df["score"] = pd.to_numeric(df["score"], errors='coerce')
    
    df["score"] = df["score"].fillna(df["score"].mean())
    df= df.dropna(subset =["name"])
    
    df = df.drop_duplicates()
    
    return df
    
    
def add_features(df):
    df["passed"] = df["score"]>=50   
    
    return df



df = load("students_messy.csv")

df = clean(df)

df = add_features(df)

df.to_csv("students_clean.csv", index=False)

print("Pipeline completed successfully!")
print(df)