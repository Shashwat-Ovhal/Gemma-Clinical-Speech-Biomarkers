import os
import PyPDF2

assets_dir = r"c:\Users\Shashwat\OneDrive\Desktop\Gemma-Clinical-Speech-Biomarkers\assets"
pdfs = [f for f in os.listdir(assets_dir) if f.endswith('.pdf')]

with open('pdf_summaries.txt', 'w', encoding='utf-8') as out_f:
    for pdf in pdfs:
        out_f.write(f"=== {pdf} ===\n")
        pdf_path = os.path.join(assets_dir, pdf)
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                # Read first 3 pages
                for i in range(min(3, len(reader.pages))):
                    text += reader.pages[i].extract_text() + "\n"
                out_f.write(text[:3000] + "\n\n" + "="*50 + "\n\n")
        except Exception as e:
            out_f.write(f"Error reading {pdf}: {e}\n\n")
