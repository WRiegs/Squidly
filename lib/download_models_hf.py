from huggingface_hub import snapshot_download
import shutil
import os

base_dir = os.getcwd()
squidly_model_dir = os.path.join(base_dir, "squidly", "models")
download_dir = snapshot_download(
    repo_id="WillRieger/Squidly",
    local_dir=os.path.join(base_dir, "squidly"),
    local_dir_use_symlinks=False
)
print(f"Models downloaded to: {squidly_model_dir}")
