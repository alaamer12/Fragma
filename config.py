import shutil
import os
from tqdm.auto import tqdm
import nltk
import pandas as pd

def download_ds(url):
  import kagglehub

  # Download latest version
  path = kagglehub.dataset_download(url)

  print("Path to dataset files:", path)
  return path

def mount_drive():
  from google.colab import drive
  """Uploads the dataset_fragments.csv file to Google Drive."""
  drive.mount('/content/drive')

def move_to_nlp_drive(file, path):
  if not os.path.exists(path):
    raise ValueError("Path does not exist")

  shutil.move(file, path)

def safe_download_nltk_resources():
    packages = ['stopwords', 'punkt', "punkt_tab"]
    for package in tqdm(packages, unit="pak", desc="Downloading resources", leave=True):
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Failed to download {package}: {e}")

def count_fragments(df):
    count_t = df['is_fragment'].sum() # True Count
    count_f = len(df) - count_t # False Count
    return count_t, count_f

def get_df(path, **kwargs):
  return pd.read_csv(path, **kwargs)

def sync_colab_workspace(files: list, sync_strategy: Literal["clone", "upload", "copy"] = "copy", **kwargs):
    """Ensure all modules and files exist in Google Colab."""
    from google.colab import files, drive

    script_files = files

    if sync_strategy == "copy":
        drive_path = "/content/drive"
        if not os.path.exists(drive_path):
            drive.mount()

        nlp_project_path = kwargs.get("nlp")

        for file in tqdm(script_files, leave=False):
            src = os.path.join(nlp_project_path, file)
            dst = os.path.join("/content", file)
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"Warning: {file} not found in Drive path.")

    elif sync_strategy == "upload":
        uploaded = files.upload()
        for name in uploaded.keys():
            print(f"Uploaded: {name}")

    elif sync_strategy == "clone":
        repo_url = kwargs.get("repo_url")
        project_name = kwargs.get("project_name")

        repo_dir = f"/content/{project_name}"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        subprocess.run(["git", "clone", repo_url], check=True)

        for file in script_files:
            src = os.path.join(repo_dir, file)
            dst = os.path.join("/content", file)
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"Warning: {file} not found in cloned repo.")

    else:
        raise ValueError("Invalid sync_strategy. Choose from 'copy', 'upload', or 'clone'.")

    def verify_files():
        print("\nVerifying required files in /content:")
        missing = []
        for file in script_files:
            if not os.path.exists(os.path.join("/content", file)):
                missing.append(file)
        if missing:
            print("Missing files:", missing)
        else:
            print("✅ All files are present.")

    verify_files()

def download_file(raw_url, output):
  import requests

  output_filename = output

  # Perform the download
  response = requests.get(raw_url)

  # Check if the request was successful
  if response.status_code == 200:
      with open(output_filename, 'wb') as f:
          f.write(response.content)
      print(f"File downloaded successfully as '{output_filename}'")
  else:
      print(f"Failed to download file. Status code: {response.status_code}")

def check_replace_nulls(df: pd.DataFrame, text_column: str = 'Processed Text') -> pd.DataFrame:
    null_count = df[text_column].isnull().sum()
    if null_count > 0:
        info(f"Found {null_count} null values in '{text_column}' column. Replacing with empty strings.")
        df[text_column] = df[text_column].fillna('')
    else:
        success(f"No null values found in '{text_column}' column.")
    return df

def check_nltk_resources():
    resources = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt'
    }

    for name, path in tqdm(resources.items(), unit="resource", desc="Checking resources", leave=True):
        try:
            nltk.data.find(path)
            print(f"[✓] {name} is available.")
        except LookupError:
            print(f"[✗] {name} is NOT available.")