import os
import logging
from typing import Union
from supabase import create_client, Client

class SupabaseService:
    def __init__(self, supabase_url: str, supabase_anon_key: str, cache_dir: str):
        self.supabase: Client = create_client(supabase_url, supabase_anon_key)
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def download_assets(self, assets: list[str]):
        """Download the assets required by the workflow from Supabase Storage."""
        for asset_id in assets:
            try:
                # Fetch asset metadata to get the storage_path
                response = self.supabase.from_('assets').select('storage_path').eq('id', asset_id).single()
                if response.error:
                    logging.error(f"Error fetching asset {asset_id} metadata: {response.error.message}")
                    continue

                storage_path = response.data['storage_path']
                file_name = os.path.basename(storage_path)
                asset_local_path = os.path.join(self.cache_dir, file_name)

                if not os.path.exists(asset_local_path):
                    logging.info(f"Downloading asset: {file_name} from {storage_path}")
                    # Download the file from Supabase Storage
                    download_response = self.supabase.storage.from_('dgn-assets').download(storage_path)
                    if download_response.error:
                        logging.error(f"Error downloading asset {file_name}: {download_response.error.message}")
                        continue

                    with open(asset_local_path, 'wb') as f:
                        f.write(download_response.data)
                    logging.info(f"Asset {file_name} downloaded to {asset_local_path}")
                else:
                    logging.info(f"Asset {file_name} already exists in cache.")
            except Exception as e:
                logging.error(f"An error occurred during asset download for {asset_id}: {e}")

    def upload_output(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the output file to Supabase Storage and return the storage path."""
        try:
            with open(file_path, 'rb') as f:
                file_name = os.path.basename(file_path)
                storage_path = f"{job_id}/{file_name}"
                response = self.supabase.storage.from_('scene-videos').upload(storage_path, f.read(), {'content-type': 'video/mp4'})
                logging.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {response}")
                if response.path:
                    logging.info(f"File {file_name} uploaded successfully to {response.path}.")
                    return response.path
                else:
                    logging.error(f"Unexpected response from Supabase upload for file {file_name}: {response}")
                    return None
        except Exception as e:
            logging.error(f"Could not upload file {file_path} to Supabase: {e}")
            return None