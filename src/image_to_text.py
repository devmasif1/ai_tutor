import os
import pdfplumber
from PIL import Image
import pytesseract

def ocr_image_to_md(image_path, md_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_pdf(pdf_path):
    md_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                pil_image = page.to_image(resolution=300).original
                text = pytesseract.image_to_string(pil_image)
                md_lines.append(f"## Page {i+1}\n{text}\n")
    if md_lines:
        md_path = os.path.splitext(pdf_path)[0] + ".md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(md_lines)
        os.remove(pdf_path)

def process_image(image_path):
    md_path = os.path.splitext(image_path)[0] + ".md"
    ocr_image_to_md(image_path, md_path)
    os.remove(image_path)


def process_folder(folder):
    for root, dirs, files in os.walk(folder):
        print("Processing folder:", root, "Subfolders:", dirs, "Files:", files)
        for filename in files:
            path = os.path.join(root, filename)
            if filename.lower().endswith(".pdf"):
                process_pdf(path)
            elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")):
                process_image(path)



if __name__ == "__main__":
    folder = "E:\\New folder\\RagProject\\knowledge_base\\Semester_4"
    process_folder(folder)