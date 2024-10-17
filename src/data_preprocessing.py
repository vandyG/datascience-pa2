# %% [markdown]
# # Data Engineering

# %%
import missingno as msno
import pandas as pd
from numpy import nan, sin, cos, pi
from sklearn.preprocessing import LabelEncoder

# %%
data = pd.read_csv("../docs/dataset_DT.csv")
data.dtypes

# %%
data["pdays"].unique()

# %%
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].replace([nan, "", " ", "unknown"], pd.NA)

data["pdays"] = data["pdays"].replace([-1], [float(nan)])


# %%
msno.matrix(data.sample(250))

# %%
msno.bar(data)

# %%
msno.heatmap(data)

# %%
msno.dendrogram(data)

# %%
percent_missing = data.isnull().sum() * 100 / len(data)

print(percent_missing)

# %%
data.dropna(subset=["job"], inplace=True)

# %%
# Drop records where poutcome is null and pdays is not. This is likely a sampling error.
data = data[~((data["poutcome"].isnull()) & (data["pdays"].notnull()))]

# %%
# Creating day to int from float
data["day"] = pd.to_numeric(data["day"], errors="coerce").fillna(0).astype(int)
data["day"]


# %%
# Define a function to handle day and month data, and add the year 2024
def create_date(row):
    """Creates a date from the 'day' and 'month' columns in a DataFrame row, assuming the year is 2024.

    The function handles potential errors in day and month values and returns NaT (Not a Time)
    if a valid date cannot be created.

    Args:
        row (pandas.Series): A row from a Pandas DataFrame containing 'day' and 'month' columns.

    Returns:
        pandas.Timestamp or pandas.NaT:
            - A pandas Timestamp representing the created date if successful.
            - NaT (Not a Time) if either 'day' or 'month' is missing or if there's an error
              converting the values to a date.
    """
    # Only create date if both day and month are available
    if pd.notna(row["day"]) and pd.notna(row["month"]):
        try:
            # Ensure that day is integer if possible
            day = int(row["day"])
            # Use the month from the row and assume 2024 as the year
            return pd.to_datetime(f'2024-{row["month"]}-{day}', format="%Y-%b-%d", errors="coerce")
        except Exception:
            return pd.NaT  # Return a NaT (Not a Time) if any error occurs
    else:
        return pd.NaT  # Return NaT if either day or month is missing


# Apply the function to each row
data["date"] = data.apply(create_date, axis=1)

# Fill forward the missing dates by propagating the previous non-null date
data["date"] = data["date"].ffill()

# %%
data = data.drop(["day", "month"], axis=1)
data["day_of_week"] = data["date"].dt.day_of_week
data["month"] = data["date"].dt.month
data = data.drop("date", axis=1)

# %%
# Sine and cosine transformation for months (1-12)
data["month_sin"] = sin(2 * pi * data["month"] / 12)
data["month_cos"] = cos(2 * pi * data["month"] / 12)

# Sine and cosine transformation for day of the week (0-6)
data["day_of_week_sin"] = sin(2 * pi * data["day_of_week"] / 7)
data["day_of_week_cos"] = cos(2 * pi * data["day_of_week"] / 7)

# %%
data.head(25)

# %%
# Define the custom order for education
custom_order = ["primary", "secondary", "tertiary"]

# Create a mapping from each category to an integer based on the order
education_map = {category: idx for idx, category in enumerate(custom_order)}

# Apply the mapping to the column (including NaN handling)
data["education"] = data["education"].apply(
    lambda x: education_map[x] if x in education_map else None
)

# %%
data["pdays"] = data["pdays"].fillna(-1)
data["poutcome"] = data["poutcome"].fillna("unknown")


# %%
def one_hot_encode_column(df, column, treat_missing_as_category=False):
    """One-hot encodes a specified column in the DataFrame.

    Optionally, treats missing values (NaN) as a separate category.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the column to one-hot encode.
    treat_missing_as_category (bool): If True, treat NaN as a separate category.

    Returns:
    pd.DataFrame: A DataFrame with the one-hot encoded column.
    """
    # If missing values should be treated as a separate category, fill NaNs with a placeholder
    if treat_missing_as_category:
        df[column] = df[column].fillna("unknown")

    # One-hot encode the column
    df_encoded = pd.get_dummies(df, columns=[column], prefix=column)

    return df_encoded


data = one_hot_encode_column(data, "job")
data = one_hot_encode_column(data, "marital")
data = one_hot_encode_column(data, "default")
data = one_hot_encode_column(data, "housing")
data = one_hot_encode_column(data, "loan")
data = one_hot_encode_column(data, "contact", True)
data = one_hot_encode_column(data, "poutcome")

# %%
# Apply the mapping to the column (including NaN handling)
data["y"] = data["y"].map({"yes": 1, "no": 0})

# %%
data.to_csv("../docs/out.csv")

# %%
from sklearn.model_selection import train_test_split

X = data.drop(["y"], axis=1)
y = data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8, test_size=0.3)

# %%
