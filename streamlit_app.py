#!/usr/bin/env python3
"""
start the Streamlit interface.
"""

import subprocess
import sys
from pathlib import Path

def main():

    app_path = Path(__file__).parent / "aci_py" / "gui" / "app.py"

    if not app_path.exists():
        print("Error: No GUI files found!")
        return 1

    try:
        # Launch streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    except KeyboardInterrupt:
        print("\n\nGUI closed.")
    except Exception as e:
        print(f"\nError launching GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())