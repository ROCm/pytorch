import pandas as pd
from pathlib import Path
import sys
import subprocess

def export_data_df(
    data_df: pd.DataFrame,
    output_folder_path: Path,
    output_filename: str,
    output_table_format: list = [".xlsx", ".csv"],
    suffix: str = "_summary_statistics",
    verbose: int = 0,
) -> None:
    """
    Exports a pandas DataFrame to one or more file formats (.xlsx, .csv) in the specified output directory.

    Args:
        data_df (pd.DataFrame): The DataFrame containing data to export.
        output_folder_path (Path): The directory where the output file(s) will be saved.
        output_filename (str): The base name of the output file.
        output_table_format (list, optional): A list of desired file extensions (e.g. [".xlsx", ".csv"]).
        suffix (str, optional): Suffix added to the output filename before the extension. Defaults to "_summary_statistics".
        verbose (int, optional): If > 0, prints additional information during processing. Defaults to 0.

    Returns:
        None
    """
    if verbose:
        print(f"Exporting data to {output_folder_path}")
    if verbose > 3:
        print(f"Data: {data_df}")

    data_df = data_df.round(2)
    
    for output_table_format in output_table_format:
        if output_table_format == ".xlsx":
            output_path = output_folder_path.joinpath(
                output_filename + suffix
            ).with_suffix(".xlsx")
            if verbose:
                print(f"Exporting summary statistics to {output_path}")

            data_df.to_excel(output_path, index=False)
        elif output_table_format == ".csv":
            output_path = output_folder_path.joinpath(
                output_filename + suffix
            ).with_suffix(".csv")
            if verbose:
                print(f"Exporting summary statistics to {output_path}")
            data_df.to_csv(output_path, index=False)

def request_install(package_name):
    """
    Prompts the user to install a Python package via pip. If the user agrees, attempts installation.
    Exits the program if the user declines or if installation fails.

    Args:
        package_name (str): The name of the package to install.

    Returns:
        None

    Side Effects:
        - Prompts the user for input.
        - May install a package using pip.
        - Exits the program (calls sys.exit(1)) if the user declines or installation fails.
    """
    choice = input(f"Do you want to install '{package_name}' via pip? [y/N]: ").strip().lower()
    if choice == 'y':
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        except subprocess.CalledProcessError:
            print(f"Failed to install '{package_name}'. Please install it manually. Exiting.")
            sys.exit(1)
    else:
        print(f"Skipping installation of '{package_name}' and exiting.")
        sys.exit(1)