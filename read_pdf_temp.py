
import sys

def try_pypdf2(path):
    try:
        import PyPDF2
        print("Using PyPDF2...")
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except ImportError:
        print("PyPDF2 not installed.")
        return None
    except Exception as e:
        print(f"PyPDF2 failed: {e}")
        return None

def try_pdfminer(path):
    try:
        from pdfminer.high_level import extract_text
        print("Using pdfminer...")
        return extract_text(path)
    except ImportError:
        print("pdfminer not installed.")
        return None
    except Exception as e:
        print(f"pdfminer failed: {e}")
        return None

def main():
    path = "/Users/wuleyan/Desktop/dachuang/whuphy-attention/WLY/大创申报书 (4).pdf"
    text = try_pypdf2(path)
    if text is None:
        text = try_pdfminer(path)
    
    if text:
        print("--- EXTRACTED TEXT ---")
        print(text[:5000]) # Print first 5000 chars
        print("--- END OF EXTRACTED TEXT ---")
    else:
        print("Could not extract text with available libraries.")

if __name__ == "__main__":
    main()
