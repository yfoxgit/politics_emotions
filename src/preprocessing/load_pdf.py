from pathlib import Path
import fitz
import pandas as pd


# extracts repo root path
def get_repo_root() -> Path:
    '''
    Extracts the path to the repo.

    Returns:
    Path: Path to the repo
    '''
    try:  # for .py files
        script_path = Path(__file__).resolve()
        return script_path.parent.parent.parent
    except NameError:  # for .ipynb files
        current_dir = Path.cwd().resolve()
        return current_dir


# extracts PDF text, formats into dataframe
def extract_pdf_text(folderpath: str) -> pd.DataFrame:
    '''
    Extracts text from PDFs.
    Assumes PDFs are named "department_year_authortype.pdf".

    Parameters:
        folderpath (str): Folder containing PDFs.

    Returns:
        pd.DataFrame: Columns -
            department,
            year,
            author_type,
            text
    '''
    folder = Path(folderpath)
    records = []

    for file in folder.glob("*.pdf"):
        # Extract metadata from filename
        name = file.stem  # Remove ".pdf"
        parts = name.split("_")  # Split by filename
        department = parts[0]
        year = int(parts[1])
        authortype = parts[2]

        # Open the PDF with PyMuPDF
        doc = fitz.open(file)
        full_text = []  # To accumulate text for the entire document

        # Iterate through each page in the PDF
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            # Add the page's text to the full document text
            full_text.append(page_text)

        # Join all the page text into a single string for the entire document
        document_text = "\n".join(full_text)

        # Add this document's data to the records list
        records.append({
            "department": department,
            "year": year,
            "author_type": authortype,
            "text": document_text  # Use the accumulated text from all pages
        })

        # Close the document after processing
        doc.close()

    # Check if all records were extracted (debugging check)
    if len(records) == 71:
        print("All PDFs extracted.")
    else:
        print(f"{len(records)} PDFs extracted.")

    # Return the DataFrame containing the records
    return pd.DataFrame(records)
