from config import CARLA_INSTALLATION_PATH, CARLA_GLASS_PATH, ASSET_PATH

import os
import os.path
import subprocess

if not CARLA_INSTALLATION_PATH:
    raise ValueError("⚠️ CARLA_INSTALLATION_PATH is not set. Please set it in the config.py file.")

if not os.path.exists(CARLA_INSTALLATION_PATH):
    raise ValueError(f"⚠️ CARLA installation path {CARLA_INSTALLATION_PATH} does not exist. Please check the path in the config.py file.")

carla_script_path = os.path.join(CARLA_INSTALLATION_PATH, "CarlaUE4.sh")
if not os.path.exists(carla_script_path):
    raise ValueError(f"⚠️ CarlaUE4.sh script does not exist at {carla_script_path}. Please check the path in the config.py file.")

glass_file_path = os.path.join(CARLA_INSTALLATION_PATH, CARLA_GLASS_PATH)
if not os.path.exists(glass_file_path):
    raise ValueError(f"⚠️ CARLA Glass file does not exist. Please check the path in the config.py file.")

# save current Glass file from glass_file_path (uasset file)
with open(glass_file_path, 'rb') as f:
    glass_data = f.read()

# save the file to assets/Carla/Glass.uasset.original
glass_file_basename = os.path.basename(CARLA_GLASS_PATH)
save_path = os.path.join(ASSET_PATH,"Carla", f"{glass_file_basename}.original")

# does the .original file already exist?
print(save_path)
if os.path.exists(save_path):
    print(f"⚠️ The original glass file already exists at {save_path}. Setting this file as the original glass file.")

    with open(save_path, 'rb') as f:
        glass_data = f.read()
else:
    # save glass_data to save_path
    
    with open(save_path, 'wb') as f:
        f.write(glass_data)

override_glass_file_path = os.path.join(ASSET_PATH, "Carla", glass_file_basename)

with open(override_glass_file_path, 'rb') as f:
    override_glass_data = f.read()

# overide glass_file_path
with open(glass_file_path, 'wb') as f:
    f.write(override_glass_data)

# now run carla using the CarlaUE4.sh script in this terminal
# if the .sh should crash or finish, print hello world
try:
    print("✅ Starting CARLA with Low Quality...")
    
    # ADDED "-quality-level=Low" here
    #result = subprocess.run([carla_script_path, "-quality-level=Low"], check=True)
    result = subprocess.run([carla_script_path], check=True)
    print("🏁 CARLA finished with return code:", result.returncode)
except subprocess.CalledProcessError as e:
    print("⚠️ CARLA script failed with error:", e)
except Exception as e:
        print("⚠️ An unexpected error occurred:", e)
finally:
    print("Reverting changes")

    # revert the glass file to the original
    with open(glass_file_path, 'wb') as f:
        f.write(glass_data)