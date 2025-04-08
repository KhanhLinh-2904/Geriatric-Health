import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns

def load_dataset(file_path):
    # Load dataset
    df = pd.read_excel(file_path)

    # Remove the 'Favourable neurological outcome' feature if it exists
    df = df.drop(columns=["Favourable neurological outcome"], errors="ignore")

    # Convert non-numeric (word) values to NaN
    df = df.applymap(lambda x: np.nan if isinstance(x, str) and not str(x).replace('.', '', 1).isdigit() else x)

    # Label column
    label_col = "Survival to hospital discharge"

    # Count label occurrences
    print('Label (0):', (df[label_col] == 0).sum())
    print('Label (1):', (df[label_col] == 1).sum())
    print('Label (N):', df[label_col].isna().sum())

    # Drop rows where the label is NaN
    df = df.dropna(subset=[label_col])

    return df

# Feature statistic
def feature_statistic(df):
    feature_info = []

    for column in df.columns:
        unique_counts = df[column].nunique(dropna=True)
        missing_rate = df[column].isna().mean()
        median_value = df[column].median() if pd.api.types.is_numeric_dtype(df[column]) else None
        mean_value = df[column].mean() if pd.api.types.is_numeric_dtype(df[column]) else None
        mode_value = df[column].mode().values.tolist()

        if unique_counts == 1:
            feature_type = 'One_Value'
        elif unique_counts > 10 and pd.api.types.is_numeric_dtype(df[column]):
            feature_type = 'Continuous'
        else:
            feature_type = 'Categorical'

        feature_info.append({
            "Feature": column,
            "Type": feature_type,
            "Unique_Values": unique_counts,
            "Missing_Rate": missing_rate,
            "Mean": mean_value,
            "Median": median_value,
            "Mode": mode_value
        })
    print("Type : ", type(feature_info))
    return feature_info

def plot_feature_summary(feature_df):
    # Convert dictionary or list-of-dicts to DataFrame
    df = pd.DataFrame(feature_df)

    # Plot 1: Feature type count
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Type', palette='Set2')
    plt.title("Feature Type Distribution")
    plt.xlabel("Feature Type")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 2: Missing value rate (exclude zero)
    missing_df = df[df['Missing_Rate'] > 0].sort_values('Missing_Rate', ascending=False)
    if not missing_df.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=missing_df, x='Feature', y='Missing_Rate')
        plt.title("Missing Value Rate per Feature")
        plt.ylabel("Missing Rate")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # Plot 3: Median value per feature (exclude zero)
    median_df = df[df['Median'] != 0].sort_values('Median', ascending=False)
    if not median_df.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=median_df, x='Feature', y='Median')
        plt.title("Median Value per Feature")
        plt.ylabel("Median")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # Plot 4: Mean value per feature (exclude zero)
    mean_df = df[df['Mean'] != 0].sort_values('Mean', ascending=False)
    if not mean_df.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=mean_df, x='Feature', y='Mean')
        plt.title("Mean Value per Feature")
        plt.ylabel("Mean")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # Plot 5: Mode value per feature (exclude zero)
    df['First_Mode'] = df['Mode'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    mode_df = df[df['First_Mode'] != 0].sort_values('First_Mode', ascending=False)
    if not mode_df.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=mode_df, x='Feature', y='First_Mode')
        plt.title("Mode Value (First) per Feature")
        plt.ylabel("Mode")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

# Fill NaN
def fill_missing_values(DataSet, feature_info):
    
    for item in feature_info:
        if item is None:
            continue
        name = item['Feature']
        if item['Type'] == 'Continuous':
            DataSet[name].fillna(value=int(item['Mean']), inplace=True)
        elif item['Type'] == 'Categorical':
            DataSet[name].fillna(value=int(item['Mode'][0]), inplace=True)
        elif item['Type'] == 'One_Value':
            DataSet.drop(columns=[name], inplace=True)
 
    print('\n# After FillNaN :', DataSet.isna().sum().sum())
    return DataSet

def data_split_cross_validation(DataSet, output_dir="splits", n_splits=5, random_seed=99):
    label = 'Survival to hospital discharge'
    X = DataSet.drop(columns=label).to_numpy()
    Y = DataSet[label].to_numpy()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_seed)

    for fold, (train_val_idx, test_idx) in enumerate(sss.split(X, Y), 1):
        # Split into train_val and test
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        Y_train_val, Y_test = Y[train_val_idx], Y[test_idx]

        # Further split train_val into train and validation
        val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1111, random_state=random_seed)  # 0.1111 * 0.9 ≈ 0.1 overall
        train_idx, val_idx = next(val_split.split(X_train_val, Y_train_val))
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        Y_train, Y_val = Y_train_val[train_idx], Y_train_val[val_idx]

        # Print stats
        print(f"Fold {fold}:")
        print(f"  Train     : {X_train.shape} -- {np.sum(Y_train == 1)} / {np.sum(Y_train == 0)}")
        print(f"  Val       : {X_val.shape} -- {np.sum(Y_val == 1)} / {np.sum(Y_val == 0)}")
        print(f"  Test      : {X_test.shape} -- {np.sum(Y_test == 1)} / {np.sum(Y_test == 0)}")

        # Save to .npz file
        np.savez(os.path.join(output_dir, f"fold_{fold}.npz"),
                 x_train=X_train, y_train=Y_train,
                 x_val=X_val, y_val=Y_val,
                 x_test=X_test, y_test=Y_test)

    print(f"\n Saved {n_splits} stratified split files to '{output_dir}/'")

# Example usage
if __name__ == "__main__":
    file_path = "pone.0166148.s001.xlsx"
    df = load_dataset(file_path)
    feature_info = feature_statistic(df)
    # plot_feature_summary(feature_info)
    Dataset = fill_missing_values(df, feature_info)
    data_split_cross_validation(Dataset)


    


