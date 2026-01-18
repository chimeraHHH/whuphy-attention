
import pypdf
import sys

file_path = r"d:\Github hanjia\whuphy-attention\BJQ\2026年大学生创新创业训练计划项目申报书12.18.pdf"

try:
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Write to a text file
    with open("paper_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Content written to paper_content.txt")
except Exception as e:
    print(f"Error reading PDF: {e}")
