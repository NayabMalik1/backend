import os
import zipfile
from typing import List

def extract_dex_files_from_apk(apk_path: str, extract_dir: str) -> List[str]:
    """
    Extract all .dex files from an APK into extract_dir.
    Returns list of extracted dex file paths.
    """

    if not os.path.isfile(apk_path):
        raise FileNotFoundError(f"APK file not found: {apk_path}")

    os.makedirs(extract_dir, exist_ok=True)
    extracted_dex_paths: List[str] = []

    try:
        with zipfile.ZipFile(apk_path, "r") as apk_zip:
            for member in apk_zip.namelist():
                if member.lower().endswith(".dex"):
                    apk_zip.extract(member, extract_dir)
                    extracted_path = os.path.join(extract_dir, member)
                    extracted_dex_paths.append(extracted_path)
    except zipfile.BadZipFile as e:
        raise Exception(f"Invalid or corrupted APK: {apk_path}") from e
    except Exception as e:
        raise Exception(f"Failed to extract APK: {apk_path}") from e

    if not extracted_dex_paths:
        raise Exception(f"No .dex file found in APK: {apk_path}")

    return extracted_dex_paths

def extract_primary_dex_from_apk(apk_path: str, extract_dir: str) -> str:
    """
    Extract only primary classes.dex if present.
    Otherwise return the first dex file found.
    """

    dex_files = extract_dex_files_from_apk(apk_path, extract_dir)

    for dex_path in dex_files:
        if os.path.basename(dex_path).lower() == "classes.dex":
            return dex_path

    return dex_files[0]
