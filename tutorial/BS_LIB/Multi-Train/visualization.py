import os

def get_newest_folder(base_folder, label):
    base_dirs = os.listdir(base_folder)
    max_base_date  = ""
    for dirs in base_dirs:
        if label in dirs:
            max_base_date = dirs
            break
    
    for dirs in base_dirs:
        if label in dirs and dirs > max_base_date:
            max_base_date = dirs

    return os.path.join(base_folder, max_base_date)