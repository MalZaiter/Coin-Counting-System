#!/usr/bin/env python3
"""Wrapper to run tests and document results of extraction."""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_documented"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Run test
result = subprocess.run([sys.executable, "-m", "tester.test_feature_extract"], 
                       capture_output=True, text=True, cwd=str(ROOT))

# Create documentation
doc_file = OUT_DIR / "feature_extraction_results.txt"
with open(doc_file, "w", encoding="utf-8") as f:
    f.write("="*70 + "\n")
    f.write("FEATURE EXTRACTION TEST RESULTS\n")
    f.write("="*70 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("\n")
    
    f.write("SCRIPT OUTPUT:\n")
    f.write("-"*70 + "\n")
    f.write(result.stdout)
    if result.stderr:
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    f.write("\n")
    f.write("-"*70 + "\n")
    f.write("\n")
    
    f.write("EXECUTION SUMMARY:\n")
    f.write("-"*70 + "\n")
    f.write(f"Return Code: {result.returncode}\n")
    f.write(f"Status: {'SUCCESS' if result.returncode == 0 else 'FAILED'}\n")
    f.write("\n")
    
    f.write("OUTPUT ARTIFACTS:\n")
    f.write("-"*70 + "\n")
    features_root = ROOT / "features_extracted"
    if features_root.exists():
        for item in sorted(features_root.iterdir()):
            if item.is_file() and item.suffix == ".csv":
                f.write(f"- {item.relative_to(ROOT)}\n")
        for subfolder in sorted(features_root.iterdir()):
            if subfolder.is_dir():
                jpg_count = len(list(subfolder.glob("*_annotated.jpg")))
                f.write(f"- {subfolder.relative_to(ROOT)}/ ({jpg_count} annotated images)\n")
    f.write("\n")
    
    f.write("NOTES:\n")
    f.write("-"*70 + "\n")
    f.write("- Feature extraction validates feature vector length and NaN values\n")
    f.write("- Annotated images show detected circles with center points and labels\n")
    f.write("- CSV files contain all extracted features for each detected circle\n")
    f.write("- Both raw and complex_tests folders are processed\n")

print(f"Results documented in: {doc_file}")
sys.exit(result.returncode)
