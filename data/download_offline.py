import os
from datasets import load_dataset

DEST_DIR = os.path.join(os.path.dirname(__file__), 'cnn_dailymail_offline')

def download_and_save():
    print(f"Downloading/loading full CNN/DailyMail 3.0.0 from Hugging Face...")
    ds = load_dataset('abisee/cnn_dailymail', '3.0.0')
    print(f"Saving dataset offline to: {DEST_DIR}")
    ds.save_to_disk(DEST_DIR)
    print("Done! The dataset is now completely offline.")

if __name__ == '__main__':
    download_and_save()
