# Step 13 - Compile Interactive Demo
# Generates the standalone HTML browser simulations (both the technical dashboard and the animated farm view).

import sys, json, re
sys.path.insert(0, ".")
from pathlib import Path

def main():
    print("Building standalone interactive simulations...")
    import subprocess
    # Run scratch generator to update interactive_farm.html
    cmd = [sys.executable, "C:/Users/aadib/.gemini/antigravity/brain/d9264509-6a61-444d-b31b-9a008f3e07d9/scratch/generate_interactive_farm.py"]
    subprocess.run(cmd, check=True)
    print("Demo files ready in outputs/demo.html and outputs/interactive_farm.html")

if __name__ == "__main__":
    main()
