"""
DermaStack-XAI Dataset Acquisition and Ingestion Helper.
Provides safe, configurable acquisition of the HAM10000 dataset
from Harvard Dataverse, with official MD5 verification, resumable streaming,
programmatic disk space checks, and secure path-traversal-free extraction.
"""

import argparse
import hashlib
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from src.config import DEFAULT_CONFIG
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Download")

# Official Harvard Dataverse API files and verified checksums
# Source: Tschandl et al., Scientific Data (2018), Harvard Dataverse doi:10.7910/DVN/DBW86T
DATAVERSE_FILES: List[Dict[str, str]] = [
    {
        "filename": "HAM10000_metadata.csv",
        "datafile_id": "4338392",
        "download_url": "https://dataverse.harvard.edu/api/access/datafile/4338392?format=original",
        "expected_md5": "8f85fb1aa29d80a2797247e434deb79d",
        "approx_size_bytes": 690218,
        "is_archive": False
    },
    {
        "filename": "HAM10000_images_part_1.zip",
        "datafile_id": "3172585",
        "download_url": "https://dataverse.harvard.edu/api/access/datafile/3172585",
        "expected_md5": "4639bfa73ab251610530a97c898e6e46",
        "approx_size_bytes": 1366522108,
        "is_archive": True,
        "extract_subdir": "HAM10000_images_part_1",
        "expected_images_count": 5000
    },
    {
        "filename": "HAM10000_images_part_2.zip",
        "datafile_id": "3172584",
        "download_url": "https://dataverse.harvard.edu/api/access/datafile/3172584",
        "expected_md5": "da43d6cc50f6613013be07e8986b384b",
        "approx_size_bytes": 1403566547,
        "is_archive": True,
        "extract_subdir": "HAM10000_images_part_2",
        "expected_images_count": 5015
    }
]


def compute_file_md5(filepath: Path, chunk_size: int = 1024 * 1024) -> str:
    """Computes the MD5 hex digest of a file incrementally without memory overhead."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_integrity(filepath: Path, expected_md5: str) -> bool:
    """Verifies that an existing file matches its expected MD5 checksum."""
    if not filepath.exists():
        return False
    actual_md5 = compute_file_md5(filepath)
    matches = actual_md5.lower() == expected_md5.lower()
    if not matches:
        logger.warning(
            f"Integrity check failed for {filepath.name}. "
            f"Expected: {expected_md5}, Actual: {actual_md5}"
        )
    return matches


def check_disk_space_sufficiency(target_dir: Path, min_required_gb: float = 6.0) -> Tuple[bool, float, float]:
    """
    Checks if target volume has sufficient free disk space.
    Returns (is_sufficient, free_gb, required_gb).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    free_gb = usage.free / (1024 ** 3)
    is_sufficient = free_gb >= min_required_gb
    return is_sufficient, round(free_gb, 2), min_required_gb


def safe_extract_zip(
    zip_path: Path,
    target_extract_dir: Path,
    overwrite: bool = False
) -> int:
    """
    Extracts a zip archive while preventing path traversal (Zip Slip vulnerability)
    and avoiding overwriting existing valid files.

    Returns:
        Count of extracted files.
    """
    target_extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_target = target_extract_dir.resolve()
    extracted_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Defense against Zip Slip path traversal
            target_file = (target_extract_dir / member.filename).resolve()
            if not target_file.is_relative_to(resolved_target):
                raise SecurityError(
                    f"Refusing to extract suspicious member with path traversal: {member.filename}"
                )

            if member.is_dir():
                target_file.mkdir(parents=True, exist_ok=True)
                continue

            # Non-destructive: skip if file already exists with identical size
            if target_file.exists() and not overwrite and target_file.stat().st_size == member.file_size:
                extracted_count += 1
                continue

            target_file.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, open(target_file, "wb") as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            extracted_count += 1

    return extracted_count


def download_stream_resumable(
    url: str,
    target_path: Path,
    expected_md5: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 5
) -> bool:
    """
    Downloads a file with resume support (HTTP Range) and progress bar.
    Includes automated retries on transient connection errors.
    Verifies MD5 upon completion.
    """
    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0

            headers = {}
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"
                logger.info(f"Attempting to resume download for {target_path.name} from byte {existing_bytes:,} (Attempt {attempt}/{max_retries})...")

            response = requests.get(url, headers=headers, stream=True, timeout=timeout)

            # 206 = Partial Content (resume succeeded); 200 = Full file (restart or resume not supported)
            if response.status_code == 206 and existing_bytes > 0:
                mode = "ab"
            elif response.status_code == 200:
                mode = "wb"
                existing_bytes = 0
            elif response.status_code == 416:
                # Range not satisfiable; likely already finished or corrupted range, restart
                logger.warning(f"HTTP 416 Range Not Satisfiable for {target_path.name}. Resetting partial file...")
                temp_path.unlink(missing_ok=True)
                existing_bytes = 0
                mode = "wb"
                response = requests.get(url, stream=True, timeout=timeout)
                if response.status_code != 200:
                    logger.error(f"Failed fresh request for {target_path.name}: HTTP {response.status_code}")
                    return False
            else:
                logger.error(f"Failed to download {target_path.name}. HTTP {response.status_code}: {response.text[:200]}")
                time.sleep(2)
                continue

            content_len = response.headers.get("Content-Length")
            total_bytes = (int(content_len) + existing_bytes) if content_len else None

            with open(temp_path, mode) as f, tqdm(
                total=total_bytes,
                initial=existing_bytes,
                unit="B",
                unit_scale=True,
                desc=target_path.name,
                dynamic_ncols=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Verify MD5 if provided
            if expected_md5:
                logger.info(f"Computing MD5 checksum for {target_path.name}...")
                actual_md5 = compute_file_md5(temp_path)
                if actual_md5.lower() != expected_md5.lower():
                    logger.error(
                        f"Checksum mismatch for {target_path.name}! "
                        f"Expected: {expected_md5}, Actual: {actual_md5}. Purging corrupted partial file."
                    )
                    temp_path.unlink(missing_ok=True)
                    return False
                logger.info(f"Checksum verified: {actual_md5}")

            # Atomically rename to final path
            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)
            logger.info(f"Successfully downloaded and verified {target_path.name}")
            return True

        except (requests.exceptions.RequestException, IOError) as e:
            logger.warning(f"Download attempt {attempt}/{max_retries} failed for {target_path.name}: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                logger.error(f"Exhausted all retries for {target_path.name}.")
                return False

    return False


def print_acquisition_audit() -> None:
    """Prints a clear inspection report of the dataset acquisition options and storage requirements."""
    total_download_bytes = sum(f["approx_size_bytes"] for f in DATAVERSE_FILES)
    total_download_gb = total_download_bytes / (1024 ** 3)
    approx_extracted_gb = 2.8

    is_safe, free_gb, min_gb = check_disk_space_sufficiency(DEFAULT_CONFIG.paths.raw_data_dir, min_required_gb=6.0)

    print("=" * 80)
    print("   DERMASTACK-XAI: DATASET ACQUISITION SPECIFICATION & AUDIT")
    print("=" * 80)
    print("\n1. PRIMARY OFFICIAL SOURCE:")
    print("   Publisher: Harvard Dataverse (ViDIR Group, Medical University of Vienna)")
    print("   Citation:  Tschandl et al., Scientific Data 5, 180161 (2018)")
    print("   DOI:       doi:10.7910/DVN/DBW86T")
    print("   URL:       https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
    print("   Auth Req:  NONE (Public access, CC BY-NC 4.0 license)")

    print("\n2. EXPECTED FILES & CHECKSUMS (Harvard Dataverse):")
    for f in DATAVERSE_FILES:
        size_mb = f["approx_size_bytes"] / (1024 ** 2)
        print(f"   - {f['filename']}")
        print(f"     Direct URL: {f['download_url']}")
        print(f"     Size:       {size_mb:.2f} MB ({f['approx_size_bytes']:,} bytes)")
        print(f"     MD5:        {f['expected_md5']}")

    print("\n3. STORAGE FOOTPRINT & DISK SAFETY:")
    print(f"   Download Archive Size:       ~{total_download_gb:.2f} GB")
    print(f"   Extracted Images Footprint:  ~{approx_extracted_gb:.2f} GB (10,015 JPEGs)")
    print(f"   Peak Space Required:         ~{total_download_gb + approx_extracted_gb:.2f} GB")
    print(f"   Available Space on Drive:    {free_gb} GB (Sufficiency Check: {'PASSED' if is_safe else 'FAILED'})")

    print("\n4. SAFETY & INTEGRITY SAFEGUARDS:")
    print("   - Resumable Downloads: Supported via HTTP Range headers (HTTP 206).")
    print("   - Integrity Verification: Checked via official MD5 hashes after download.")
    print("   - Path Traversal Defense: Prevents Zip Slip vulnerabilities during extraction.")
    print("   - Non-Destructive: Skips extracting files that already exist on disk.")
    print("=" * 80)


def execute_authorized_acquisition(
    target_dir: Path,
    extract: bool = True
) -> Dict:
    """Executes dataset download and safe extraction once authorized."""
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Beginning authorized acquisition into {target_dir.resolve()}...")

    # 1. Disk Space Verification
    is_safe, free_gb, min_gb = check_disk_space_sufficiency(target_dir, min_required_gb=6.0)
    if not is_safe:
        err = f"Insufficient disk space in {target_dir}. Free: {free_gb} GB, Required: {min_gb} GB."
        logger.error(err)
        return {"success": False, "error": err}

    download_results = {}
    extracted_counts = {}

    for file_info in DATAVERSE_FILES:
        dest_path = target_dir / file_info["filename"]

        # Check if file already exists and is intact
        if dest_path.exists() and verify_file_integrity(dest_path, file_info["expected_md5"]):
            logger.info(f"File {dest_path.name} already exists with valid checksum. Skipping download.")
            download_results[file_info["filename"]] = {
                "downloaded": False,
                "already_valid": True,
                "size_bytes": dest_path.stat().st_size,
                "md5": file_info["expected_md5"]
            }
        else:
            success = download_stream_resumable(
                url=file_info["download_url"],
                target_path=dest_path,
                expected_md5=file_info["expected_md5"]
            )
            if not success:
                err = f"Acquisition halted due to error downloading {dest_path.name}."
                logger.error(err)
                return {"success": False, "error": err, "partial_results": download_results}
            download_results[file_info["filename"]] = {
                "downloaded": True,
                "already_valid": False,
                "size_bytes": dest_path.stat().st_size,
                "md5": file_info["expected_md5"]
            }

        # Extract archive if requested
        if extract and file_info.get("is_archive"):
            subdir_name = file_info.get("extract_subdir", dest_path.stem)
            extract_dir = target_dir / subdir_name
            logger.info(f"Extracting {dest_path.name} to {extract_dir} with Zip Slip protection...")
            count = safe_extract_zip(dest_path, extract_dir)
            extracted_counts[dest_path.name] = count
            logger.info(f"Extracted {count} images to {extract_dir}.")

    # Count total images extracted
    total_extracted_images = sum(extracted_counts.values())
    logger.info(f"HAM10000 acquisition complete. Total extracted images: {total_extracted_images}.")

    return {
        "success": True,
        "download_results": download_results,
        "extracted_counts": extracted_counts,
        "total_extracted_images": total_extracted_images,
        "raw_dir": str(target_dir.resolve())
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAM10000 Dataset Acquisition Helper")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_CONFIG.paths.raw_data_dir,
        help="Target raw data directory"
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        default=False,
        help="Explicit user approval to perform network acquisition"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        default=False,
        help="Download archives only, do not extract"
    )
    args = parser.parse_args()

    print_acquisition_audit()

    if not args.approve:
        print("\n[!] SAFETY HALT: Dataset acquisition requires explicit user authorization.")
        print("To authorize and initiate acquisition, run with --approve:")
        print("  py -3.12 -m src.data.download --approve\n")
        sys.exit(0)

    result = execute_authorized_acquisition(args.target_dir, extract=not args.no_extract)
    if not result.get("success"):
        print(f"\n[!] ERROR: {result.get('error')}\n")
        sys.exit(1)
    else:
        print("\n[OK] HAM10000 DATASET ACQUISITION AND EXTRACTION SUCCESSFUL!")
        print(f"Total extracted images: {result['total_extracted_images']}")
        print(f"Directory: {result['raw_dir']}\n")
