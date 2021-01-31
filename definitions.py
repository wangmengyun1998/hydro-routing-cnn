import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
print(ROOT_DIR)
# add project to the sys path
sys.path.append(ROOT_DIR)
CONFIG_DIR = os.path.join(ROOT_DIR, "example", "User")
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.ini')  # your own configuration file
if not os.path.isdir(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)
if not os.path.isfile(CONFIG_FILE):
    shutil.copy(os.path.join(ROOT_DIR, "data", "config.ini"), CONFIG_FILE)
