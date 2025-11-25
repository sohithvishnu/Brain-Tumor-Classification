import subprocess
import sys
import os
import zipfile

# -------------------------------------------------------
# 1. ENSURE PIP IS INSTALLED
# -------------------------------------------------------
def ensure_pip():
    try:
        import pip
        print("[INFO] pip is available.")
    except ImportError:
        print("[INFO] pip not found — installing using ensurepip...")
        import ensurepip
        ensurepip.bootstrap()
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

ensure_pip()

# -------------------------------------------------------
# 2. INSTALL REQUIRED PACKAGES
# -------------------------------------------------------
required_packages = [
    "gdown",
    "torch",
    "torchvision", 
    "torchaudio",
    "opencv-python", "pillow", 
    "numpy","pandas","scikit-learn","matplotlib"
]

def install_packages(packages):
    for pkg in packages:
        print(f"[INFO] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_packages(required_packages)

# -------------------------------------------------------
# 3. DOWNLOAD ZIP FILE FROM GOOGLE DRIVE
# -------------------------------------------------------
import gdown  # now installed

file_id = "1zdSSFCsAI0CV3jQRgpUlPwaYGUaBbKk-"
url = f"https://drive.google.com/uc?id={file_id}"

output_dir = "./data/"
os.makedirs(output_dir, exist_ok=True)

zip_path = os.path.join(output_dir, "dataset.zip")

print("[INFO] Downloading dataset ZIP...")
gdown.download(url, zip_path, quiet=False)

# -------------------------------------------------------
# 4. UNZIP THE DOWNLOADED FILE
# -------------------------------------------------------
print("[INFO] Unzipping dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)
# -------------------------------------------------------
# 5. FIND TRAINING AND TESTING FOLDERS ANYWHERE IN DATA/
# -------------------------------------------------------
print("[INFO] Locating Training and Testing folders...")

training_path = None
testing_path = None

for root, dirs, files in os.walk(output_dir):
    for d in dirs:
        if d.lower() == "training":
            training_path = os.path.join(root, d)
        if d.lower() == "testing":
            testing_path = os.path.join(root, d)

if not training_path or not testing_path:
    raise RuntimeError("Could not find Training or Testing folders!")

print(f"[FOUND] Training: {training_path}")
print(f"[FOUND] Testing:  {testing_path}")

# -------------------------------------------------------
# 6. MOVE THEM TO ./data/
# -------------------------------------------------------
import shutil
final_training = os.path.join(output_dir, "Training")
final_testing = os.path.join(output_dir, "Testing")

# If exist from previous runs — remove
if os.path.exists(final_training): shutil.rmtree(final_training)
if os.path.exists(final_testing): shutil.rmtree(final_testing)

shutil.move(training_path, final_training)
shutil.move(testing_path, final_testing)

print("[INFO] Training and Testing moved to ./data/")

# -------------------------------------------------------
# 7. CLEANUP: REMOVE ZIP AND NESTED DIRECTORIES
# -------------------------------------------------------
print("[INFO] Removing ZIP file...")
os.remove(zip_path)

print("[INFO] Cleaning unused nested folders...")
for item in os.listdir(output_dir):
    path = os.path.join(output_dir, item)
    if item not in ("Training", "Testing") and os.path.isdir(path):
        shutil.rmtree(path)

print("[SUCCESS] Dataset structure ready:")
print("data/")
print("   ├── Training/")
print("   └── Testing/")