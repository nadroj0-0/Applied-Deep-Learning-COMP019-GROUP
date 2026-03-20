from pathlib import Path
import pandas as pd

#Base folders
base_folder = Path("m5_processed_validation")
features_folder = base_folder / "features"
static_folder = base_folder / "static"

#Output folder
output_dir = Path("m5_processed_validation_ca3")
output_dir.mkdir(parents = True, exist_ok=True)

#Load static series info
series_info = static_folder / "validation_series_info.pkl.gz"
series_info = pd.read_pickle(series_info, compression = "gzip")

print("Loaded series info:", series_info.shape)
print("Columns:", series_info.columns.tolist())

#Filter ids that are in CA3
ca3_series = series_info[series_info["store_id"] == "CA_3"].copy()
ca3_ids = set(ca3_series["id"].astype(str))

print("Number of series in CA3:", len(ca3_ids))

#Save filtered series info
ca3_series.to_pickle(base_folder / "validation_series_info_ca3.pkl.gz", compression = "gzip")

#Find all feature batch files
batch_files = sorted(features_folder.glob("validation_features_batch*.pkl.gz"))
print("Found feature files:", len(batch_files))

if len(batch_files) == 0:
    raise FileNotFoundError("No feature files found")

#filter each batch
total_before = 0
total_after = 0

for batch_path in batch_files:
    print(f"Processing {batch_path.name}...")

    batch_df = pd.read_pickle(batch_path, compression = "gzip")
    batch_df["id"] = batch_df["id"].astype(str)
    
    before_shape = batch_df.shape[0]
    total_before += len(batch_df)
    
    #Filter to CA3 ids
    df_ca3 = batch_df[batch_df["id"].isin(ca3_ids)].copy()
    
    after_shape = df_ca3.shape[0]
    total_after += len(df_ca3)
    
    print(f"  Before: {before_shape} rows, After: {after_shape} rows")
    
    #Save filtered batch
    out_path = output_dir / batch_path.name
    df_ca3.to_pickle(out_path, compression = "gzip")

print("\nDone")
print(f"Total rows before filtering: {total_before}")
print(f"Total rows after filtering: {total_after}")
print(f"Filtered data saved to: {output_dir}")

'''
test = pd.read_pickle("m5_processed_validation_ca3/validation_features_batch_005.pkl.gz", compression = "gzip")
print(test.shape)
print(test['id'].head())
print(test.columns.tolist())
'''