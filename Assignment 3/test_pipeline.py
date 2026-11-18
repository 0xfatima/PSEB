from pipeline import load, clean

def test_clean():
    df = load("students_messy.csv")
    cleaned_df = clean(df)

    # Check that duplicate rows are removed
    assert cleaned_df.duplicated().sum() == 0

    # Check that names are not missing
    assert cleaned_df["name"].isnull().sum() == 0

    # Check that scores are not missing
    assert cleaned_df["score"].isnull().sum() == 0