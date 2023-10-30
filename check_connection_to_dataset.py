import pandas as pd

FOLDER_PATH = "/work3/s164396/data/DNS-Challenge-4/"
FILE_NAME = "dns4-datasets-files-sha1.csv"


with open(FOLDER_PATH+FILE_NAME,'r') as file:
  csvfile = pd.read_csv(file)
  print(csvfile)