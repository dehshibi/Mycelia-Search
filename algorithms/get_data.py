import subprocess
import shutil
import os
import stat

# Configuration
repo_url = "https://github.com/P-N-Suganthan/2022-SO-BO.git"
repo_name = "2022-SO-BO"
target_subfolder = "input_data"  # The folder we are looking for
destination_folder = "./input_data"


def remove_readonly(func, path, excinfo):
    """
    Helper function to fix 'Access is denied' error on Windows.
    It changes the file permission to writable before deleting.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def setup_project():
    # 1. Clone the repository
    if os.path.exists(repo_name):
        print(f"Folder '{repo_name}' already exists. Removing it to start fresh...")
        shutil.rmtree(repo_name, onerror=remove_readonly)

    print(f"Cloning {repo_name}...")
    try:
        subprocess.run(["git", "clone", repo_url], check=True)
    except subprocess.CalledProcessError:
        print("Git clone failed. Check your internet connection or Git installation.")
        return

    # 2. construct the full path to the data
    source_path = os.path.join(repo_name, target_subfolder)

    # 3. Check if the folder exists
    if os.path.exists(source_path):
        print(f"Found '{target_subfolder}'. Copying to {destination_folder}...")
        shutil.copytree(source_path, destination_folder, dirs_exist_ok=True)
        print("Success! Data copied.")
    else:
        # ERROR HANDLING: If folder is missing, show what IS there
        print(f"\n[!] Error: The folder '{target_subfolder}' was not found in the repo.")
        print(f"The repository contains the following files:")
        print("------------------------------------------------")
        for file in os.listdir(repo_name):
            print(f" - {file}")
        print("------------------------------------------------")
        print("Hint: The data you need is likely inside one of the .zip files listed above.")
        print("You may need to unzip 'Python-CEC2022.zip' or similar manually.")
        return  # Stop here, do not delete the repo so you can inspect it

    # 4. Clean up (Delete the cloned repo)
    print("Cleaning up temporary files...")
    try:
        shutil.rmtree(repo_name, onerror=remove_readonly)
        print("Cleanup complete.")
    except Exception as e:
        print(f"Warning: Could not fully clean up '{repo_name}'. Reason: {e}")


if __name__ == "__main__":
    setup_project()
