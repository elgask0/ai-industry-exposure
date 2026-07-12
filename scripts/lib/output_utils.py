#!/usr/bin/env python3
"""
Output Path Utilities

Provides helper functions for organizing output files into the new folder structure:
- outputs/analysis/figures/  # PNG plots
- outputs/analysis/txt/      # Human-readable tables
- outputs/analysis/tex/      # LaTeX tables
- outputs/analysis/          # CSV files (machine-readable)
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'


def get_output_path(table_name: str, ext: str) -> Path:
    """
    Get output path for a table/figure.

    Args:
        table_name: Name of the output file (without extension)
                   e.g., 'table1', 'pretrends_aiie_23_ew', 'placebo_density_11m'
        ext: File extension
             - 'png': goes to figures/
             - 'txt': goes to txt/
             - 'tex': goes to tex/
             - 'csv': goes to outputs/analysis/ (root level)

    Returns:
        Path object pointing to the correct subfolder

    Examples:
        >>> get_output_path('table1', 'txt')
        PosixPath('.../outputs/analysis/txt/table1.txt')

        >>> get_output_path('pretrends_aiie_23_ew', 'png')
        PosixPath('.../outputs/analysis/figures/pretrends_aiie_23_ew.png')
    """
    # Ensure OUTPUT_DIR exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if ext == 'png':
        subdir = OUTPUT_DIR / 'figures'
    elif ext in ['txt', 'tex']:
        subdir = OUTPUT_DIR / ext
    elif ext == 'csv':
        # CSV files stay at root level for compatibility
        return OUTPUT_DIR / f'{table_name}.{ext}'
    else:
        raise ValueError(f"Unknown extension: {ext}. Use 'png', 'txt', 'tex', or 'csv'.")

    # Ensure subdirectory exists
    subdir.mkdir(parents=True, exist_ok=True)

    return subdir / f'{table_name}.{ext}'


def ensure_output_structure():
    """
    Ensure all output subdirectories exist.

    Creates:
    - outputs/analysis/figures/
    - outputs/analysis/txt/
    - outputs/analysis/tex/
    """
    for subdir in ['figures', 'txt', 'tex']:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    return OUTPUT_DIR


if __name__ == '__main__':
    # Test the function
    ensure_output_structure()

    print("Output structure test:")
    print(f"  Table 1 (txt): {get_output_path('table1', 'txt')}")
    print(f"  Table 1 (tex): {get_output_path('table1', 'tex')}")
    print(f"  Pretrends (png): {get_output_path('pretrends_aiie_23_ew', 'png')}")
    print(f"  Placebo 11m (png): {get_output_path('placebo_density_11m', 'png')}")
    print(f"  CSV (csv): {get_output_path('did_table', 'csv')}")
    print(f"\nAll directories exist: {ensure_output_structure().exists()}")
