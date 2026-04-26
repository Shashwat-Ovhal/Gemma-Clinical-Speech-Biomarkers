import os
import synapseclient
from synapseclient import Synapse

def download_dataset_from_synapse():
    # Attempt to load token from environment variable, fall back to literal if not set
    auth_token = os.environ.get("SYNAPSE_AUTH_TOKEN", "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3NjQ0OTU5MCwiaWF0IjoxNzc2NDQ5NTkwLCJqdGkiOiIzNTc4NCIsInN1YiI6IjM1NzY2MjkifQ.BJ__fn73AW3CdHT3huDqcl_COEuO61dCjI70jtYh2YL_zeT-9SVf4QonVvTmjGyIF0AZnZUQqfPkSluCFZV_p6wptXTwdBDQDjIAl8EGh2sgSbNBlhc9i27bHPwUYwJWfeqT-6xHx7dYZ8aoVmA1RDJUwsgpAVQAVSr-Eo87HnGRAYKQjwlyBOHT4R-bUIsVRLc1xq86cUbA6huyzis31CCrCBbbGSES7crvjS8iqdIiTYiWzDwwBcqPByeAcEQ6FO31zxQ7pIgv9-9eUm9erpmaLS2Fys5-38GnOa929PY5Fu2vZ86MuDKyx6jpPmmwFpbDfGx7oNTw-D3Ku33YBQ")
    
    if auth_token == "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3NjQ0OTU5MCwiaWF0IjoxNzc2NDQ5NTkwLCJqdGkiOiIzNTc4NCIsInN1YiI6IjM1NzY2MjkifQ.BJ__fn73AW3CdHT3huDqcl_COEuO61dCjI70jtYh2YL_zeT-9SVf4QonVvTmjGyIF0AZnZUQqfPkSluCFZV_p6wptXTwdBDQDjIAl8EGh2sgSbNBlhc9i27bHPwUYwJWfeqT-6xHx7dYZ8aoVmA1RDJUwsgpAVQAVSr-Eo87HnGRAYKQjwlyBOHT4R-bUIsVRLc1xq86cUbA6huyzis31CCrCBbbGSES7crvjS8iqdIiTYiWzDwwBcqPByeAcEQ6FO31zxQ7pIgv9-9eUm9erpmaLS2Fys5-38GnOa929PY5Fu2vZ86MuDKyx6jpPmmwFpbDfGx7oNTw-D3Ku33YBQ":
        print("Using provided Synapse Auth Token.")
        # print("You can also export it as an environment variable: set SYNAPSE_AUTH_TOKEN=your_token_here")
    
    print("Logging into Synapse...")
    syn = synapseclient.Synapse()
    syn.login(authToken=auth_token.strip())
    
    print("Fetching download list...")
    # This retrieves the files you've added to your download list on the Synapse website
    dl_list_file_entities = syn.get_download_list()
    
    print("Download list retrieved successfully.")
    print("Number of files in download list:", len(dl_list_file_entities))
    
    # Optional: Automatically download to a specific directory
    download_dir = "./data/mpower_dataset"
    os.makedirs(download_dir, exist_ok=True)
    
    # You can loop through or trigger download based on your needs:
    # print(f"Downloading files to {download_dir}...")
    # for entity in dl_list_file_entities:
    #     syn.get(entity, downloadLocation=download_dir)
        
    print("Done!")

if __name__ == "__main__":
    download_dataset_from_synapse()
