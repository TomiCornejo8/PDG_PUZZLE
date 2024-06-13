from helpers import csvReader

folders = csvReader.load_data_from_folder_RL()

dungeon = folders[0]

print(dungeon)