from scripts.update_search_config import add_search_config

project_id = "filipegracio-ai-learning"
dataset_id = "filipegracio_fsa_restaurants"
table_id = "config_search_params"

# Coords provided by user (Lon, Lat)
new_coords = [
    (-0.1197, 51.428),
    (-0.085325, 51.482954),
    (-0.068429, 51.468681),
    (-0.105524, 51.429732),
    (-0.112850, 51.402007),
    (-0.169207, 51.400014),
    (-0.197718, 51.365842),
    (-0.239801, 51.390909),
    (-0.265550, 51.398463),
    (-0.240377, 51.408053),
    (-0.130875, 51.458639)
]

print(f"Adding {len(new_coords)} new coordinates...")
success = add_search_config(project_id, dataset_id, table_id, new_coords)

if success:
    print("Update complete.")
else:
    print("Update failed.")
