import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Connect to MySQL in Docker
DB_URI = "mysql+pymysql://mediation_user:securepassword@localhost:3306/mediation_db"
engine = create_engine(DB_URI)

# 1. Read the prepared CSV
CSV_PATH = PROJECT_ROOT / "files" / "demo_data.csv"
try:
    df_raw = pd.read_csv(CSV_PATH)
    print(f"Successfully read English CSV, total {len(df_raw)} records.")
except Exception as e:
    print(f"Failed to read CSV: {e}")
    exit()

df_db = pd.DataFrame()

# 2. Field mapping
def get_col(zh_name, en_name, default=""):
    return df_raw.get(en_name, df_raw.get(zh_name, default))

type_map = {0: 'Civil', '0': 'Civil', 1: 'Criminal', '1': 'Criminal', '民事': 'Civil', '刑事': 'Criminal'}
success_map = {1: 'Successful', '1': 'Successful', 0: 'Unsuccessful', '0': 'Unsuccessful', '成立': 'Successful', '不成立': 'Unsuccessful'}

df_db['main_type'] = get_col('案件大類型', 'Main_Category').map(type_map).fillna(get_col('案件大類型', 'Main_Category'))
df_db['prediction'] = get_col('是否成立', 'Is_Successful').map(success_map).fillna(get_col('是否成立', 'Is_Successful'))

df_db['sub_type'] = get_col('案件分類', 'Sub_Category', 'Unknown')
df_db['days'] = pd.to_numeric(get_col('調解天數', 'Days', 0), errors='coerce').fillna(0).astype(int)
df_db['rounds'] = pd.to_numeric(get_col('調解次數', 'Rounds', 1), errors='coerce').fillna(1).astype(int)
df_db['people'] = pd.to_numeric(get_col('調解人數', 'People', 2), errors='coerce').fillna(2).astype(int)

df_db['content'] = get_col('案件內容_匿名', 'Content', '')
df_db['keywords'] = get_col('關鍵詞', 'Keywords', '')
df_db['explanation'] = get_col('Dynamic_KG_推論結果', 'Explanation', '')
raw_paths = get_col('Dynamic_KG_路徑', 'KG_Paths', '')
df_db['paths'] = raw_paths.astype(str).str.replace('|', '\n', regex=False)

# 3. Clear old database and write pure English data
try:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM case_records"))
        conn.execute(text("ALTER TABLE case_records AUTO_INCREMENT = 1"))
except Exception as e:
    pass

print("Starting to write English database...")
df_db.to_sql('case_records', con=engine, if_exists='append', index=False)
print(f"Import complete! Successfully wrote {len(df_db)} pure English records!")
