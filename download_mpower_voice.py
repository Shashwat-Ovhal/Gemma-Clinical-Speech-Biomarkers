"""
download_mpower_voice.py — Level 1 Publication Dataset Builder
Fetches a balanced cohort of 60 PD + 60 HC patients from mPower (syn4993293)
with caching so partial runs can resume without re-downloading.
"""
import os
import shutil
import pandas as pd
import synapseclient

# ── Auth ────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.environ.get(
    "SYNAPSE_AUTH_TOKEN",
    "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3NjQ0OTU5MCwiaWF0IjoxNzc2NDQ5NTkwLCJqdGkiOiIzNTc4NCIsInN1YiI6IjM1NzY2MjkifQ.BJ__fn73AW3CdHT3huDqcl_COEuO61dCjI70jtYh2YL_zeT-9SVf4QonVvTmjGyIF0AZnZUQqfPkSluCFZV_p6wptXTwdBDQDjIAl8EGh2sgSbNBlhc9i27bHPwUYwJWfeqT-6xHx7dYZ8aoVmA1RDJUwsgpAVQAVSr-Eo87HnGRAYKQjwlyBOHT4R-bUIsVRLc1xq86cUbA6huyzis31CCrCBbbGSES7crvjS8iqdIiTYiWzDwwBcqPByeAcEQ6FO31zxQ7pIgv9-9eUm9erpmaLS2Fys5-38GnOa929PY5Fu2vZ86MuDKyx6jpPmmwFpbDfGx7oNTw-D3Ku33YBQ"
)

# ── Config ───────────────────────────────────────────────────────────────────
TARGET_PER_CLASS  = 60   # 60 PD + 60 HC = 120 total (Level 1 requirement)
BASE_DIR          = "./data/mpower_dataset"
CACHE_MANIFEST    = "./data/mpower_manifest.csv"   # Saves progress for resuming

VOICE_TABLE       = "syn5511444"   # Voice Activity
DEMO_TABLE        = "syn5511429"   # Demographics Survey


def login():
    syn = synapseclient.Synapse()
    syn.login(authToken=AUTH_TOKEN.strip())
    return syn


def build_health_map(syn) -> dict:
    """Returns {healthCode: {'status': 'PD'|'HC', 'age': int}} from Demographics survey."""
    print("Querying Demographics table...")
    q = syn.tableQuery(
        'SELECT healthCode, "professional-diagnosis", age '
        f'FROM {DEMO_TABLE} '
        'WHERE "professional-diagnosis" IS NOT NULL'
    )
    df = q.asDataFrame()
    df["status"] = df["professional-diagnosis"].apply(
        lambda x: "PD" if x is True else "HC"
    )
    df = df.dropna(subset=["age"]) # Ensure age is mapped for filtering
    mapping = df.set_index("healthCode")[["status", "age"]].to_dict(orient="index")
    pd_count = sum(1 for v in mapping.values() if v["status"] == "PD")
    hc_count = sum(1 for v in mapping.values() if v["status"] == "HC")
    print(f"  Demographics loaded: {pd_count} PD, {hc_count} HC health codes available (with age).")
    return mapping


def select_cohort(syn, health_map: dict) -> pd.DataFrame:
    """
    Queries Voice Activity table in batches until we have enough matched
    records per class. Uses a large LIMIT to allow for matching dropouts.
    """
    # Try to load cached manifest first (resume support)
    if os.path.exists(CACHE_MANIFEST):
        print(f"  Cache found at {CACHE_MANIFEST}. Loading existing cohort manifest...")
        df = pd.read_csv(CACHE_MANIFEST)
        pd_have = len(df[df["status"] == "PD"])
        hc_have = len(df[df["status"] == "HC"])
        print(f"  Cached: {pd_have} PD, {hc_have} HC records already logged.")
        if pd_have >= TARGET_PER_CLASS and hc_have >= TARGET_PER_CLASS:
            print("  Cohort already complete. Skipping query.")
            return df.head(TARGET_PER_CLASS * 2)
    
    print(f"Querying Voice table for {TARGET_PER_CLASS * 2}+ records to select from...")
    # Use a large LIMIT to have enough candidates to fill both classes after filtering
    q = syn.tableQuery(
        f'SELECT recordId, healthCode, "audio_audio.m4a" '
        f'FROM {VOICE_TABLE} LIMIT 5000'
    )
    df_voice = q.asDataFrame()
    df_voice = df_voice.dropna(subset=["audio_audio.m4a"])
    df_voice["status"] = df_voice["healthCode"].map(lambda x: health_map.get(x, {}).get("status"))
    df_voice["age"] = df_voice["healthCode"].map(lambda x: health_map.get(x, {}).get("age", 0))
    df_voice = df_voice.dropna(subset=["status"])

    # Age Gate: Ensure HC patients are at least 45 years old
    df_pd = df_voice[df_voice["status"] == "PD"].head(TARGET_PER_CLASS)
    df_hc = df_voice[(df_voice["status"] == "HC") & (df_voice["age"] >= 45)].head(TARGET_PER_CLASS)

    print(f"  Selected: {len(df_pd)} PD candidates, {len(df_hc)} HC candidates (aged 45+).")

    if len(df_pd) < TARGET_PER_CLASS or len(df_hc) < TARGET_PER_CLASS:
        print(f"  WARNING: Could not find {TARGET_PER_CLASS} per class. Got {len(df_pd)} PD / {len(df_hc)} HC.")
        print("  Continuing with available data...")

    df_subset = pd.concat([df_pd, df_hc]).reset_index(drop=True)
    df_subset.to_csv(CACHE_MANIFEST, index=False)
    print(f"  Manifest saved to {CACHE_MANIFEST}")
    return df_subset


def download_files(syn, df_subset: pd.DataFrame):
    """Downloads audio files with per-file caching to safely resume."""
    os.makedirs(os.path.join(BASE_DIR, "HC"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "PD"), exist_ok=True)

    # Only download records not yet on disk (caching / resume logic)
    already_on_disk = set()
    for cls in ["HC", "PD"]:
        for f in os.listdir(os.path.join(BASE_DIR, cls)):
            already_on_disk.add(os.path.splitext(f)[0])

    to_download = df_subset[~df_subset["recordId"].isin(already_on_disk)]
    print(f"\nFiles already cached: {len(already_on_disk)} | To download: {len(to_download)}")

    if to_download.empty:
        print("All files already downloaded. Skipping.")
        return

    # downloadTableColumns needs the table query result, not just the DF
    record_ids_repr = ",".join(repr(r) for r in to_download["recordId"].tolist())
    subset_query = syn.tableQuery(
        f"SELECT * FROM {VOICE_TABLE} WHERE recordId IN ({record_ids_repr})"
    )

    print("Downloading files from Synapse (may take a while)...")
    files = syn.downloadTableColumns(subset_query, ["audio_audio.m4a"])

    copied = 0
    for _, row in to_download.iterrows():
        fh_id = str(int(row["audio_audio.m4a"]))
        if fh_id in files:
            src = files[fh_id]
            dest = os.path.join(BASE_DIR, row["status"], f"{row['recordId']}.m4a")
            shutil.copy2(src, dest)
            copied += 1
            print(f"  [{copied}/{len(to_download)}] {row['recordId']}.m4a -> {row['status']}/")
        else:
            print(f"  [WARN] File handle {fh_id} not found in download result.")

    print(f"\nDownload complete. {copied} new files saved.")


def print_summary():
    pd_files = len(os.listdir(os.path.join(BASE_DIR, "PD")))
    hc_files = len(os.listdir(os.path.join(BASE_DIR, "HC")))
    print("\n-- Cohort Summary --")
    print(f"  PD recordings : {pd_files}")
    print(f"  HC recordings : {hc_files}")
    print(f"  TOTAL         : {pd_files + hc_files}")
    if pd_files >= TARGET_PER_CLASS and hc_files >= TARGET_PER_CLASS:
        print("  Level 1 requirement (N>=100, balanced) MET")
    else:
        print("  Level 1 requirement NOT yet met -- re-run to fetch more.")
    print("--------------------")


if __name__ == "__main__":
    syn = login()
    health_map = build_health_map(syn)
    df_subset  = select_cohort(syn, health_map)
    download_files(syn, df_subset)
    print_summary()
