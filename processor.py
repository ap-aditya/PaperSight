import io
from PIL import Image, UnidentifiedImageError
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.api_core import exceptions

try:
    import fitz
except ImportError as e:
    raise ImportError("PyMuPDF (fitz) is not installed. Please install it via 'pip install pymupdf'.") from e

def get_gemini_vision_response(api_key: str, prompt: str, image_bytes: bytes) -> str:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        image = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content([prompt, image])
        return getattr(response, "text", "No response text received.")
    except exceptions.ResourceExhausted:
        return "Image description failed due to Gemini API quota limit."
    except UnidentifiedImageError:
        return "Error: Unable to identify image format."
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_pdf_and_create_chunks(uploaded_file_bytes: bytes, api_key: str, analyze_images: bool, progress_bar=None) -> list[dict]:
    all_content = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    try:
        with fitz.open(stream=uploaded_file_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
            for page_num, page in enumerate(doc, start=1):
                if progress_bar:
                    progress_percentage = int((page_num / total_pages) * 100)
                    progress_bar.progress(progress_percentage, text=f"Processing page {page_num} of {total_pages}...")

                page_text = page.get_text("text")
                if page_text.strip():
                    all_content.append(f"Content from page {page_num}:\n{page_text}")

                if analyze_images:
                    images = page.get_images(full=True)
                    for img_index, img in enumerate(images, start=1):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            vision_prompt = "Describe this image from a document in detail."
                            image_desc = get_gemini_vision_response(api_key, vision_prompt, image_bytes)
                            all_content.append(f"Description of image {img_index} on page {page_num}: {image_desc}")
                        except Exception as e:
                            all_content.append(f"Error processing image {img_index} on page {page_num}: {e}")

    except Exception as e:
        raise ValueError(f"Failed to process PDF file: {str(e)}") from e

    if progress_bar:
        progress_bar.progress(100, text="Finalizing chunks...")

    combined_text = "\n\n---\n\n".join(all_content)
    chunks = text_splitter.split_text(combined_text)
    
    return [{"text": chunk} for chunk in chunks]