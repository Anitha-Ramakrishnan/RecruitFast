import os
import json
from datetime import datetime
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from docx import Document  # if used
from docx2pdf import convert
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

# ========================================
# CONFIGURATION
# ========================================
endpoint = "url"
api_key = "xyz"
model_id = "skill_resume_extractor_2"

input_folder = r"C:\Users\lvign\OneDrive\Documents\Projects\Sample Resumes\Final"  # folder containing local PDFs
pdf_folder = r"C:\Users\lvign\OneDrive\Documents\Projects\Sample Resumes\Temp_PDF"
output_folder = r"C:\Users\lvign\OneDrive\Documents\Projects\Sample Resumes\Results"
log_file = r"C:\Users\lvign\OneDrive\Documents\Projects\Sample Resumes\logs\Resume_Processing_Log.txt"

# Ensure folders exist
os.makedirs(pdf_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# ========================================
# LOGGING
# ========================================
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"[{timestamp}] {message}\n")
    print(message)

# ========================================
# STEP 5: GENERATE OPENAI EMBEDDINGS
# ========================================
from openai import AzureOpenAI
log("===== Starting embedding generation using OpenAI text-embedding-large =====")

# Folder where embeddings will be saved
embedding_output_folder = r"C:\Users\lvign\OneDrive\Documents\Projects\Sample Resumes\embeddings"
os.makedirs(embedding_output_folder, exist_ok=True)

# Initialize OpenAI client
client = AzureOpenAI(
    azure_endpoint="<endpoint_url>",
    api_key ="xyz",
    api_version="2024-12-01-preview"  # or the version your deployment uses
)
for filename in os.listdir(output_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(output_folder, filename)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Convert JSON to string (model expects text)
            text_input = json.dumps(json_data, indent=2, ensure_ascii=False)

            # Create embedding
            response = client.embeddings.create(
            model="text-embedding-3-large",   # or your deployment name
            input=text_input
)

            embedding_vector = response.data[0].embedding
            

            # Save embedding to output
            embed_filename = filename.replace(".json", "_embedding.json")
            embed_path = os.path.join(embedding_output_folder, embed_filename)

            with open(embed_path, "w", encoding="utf-8") as ef:
                json.dump({
                    "source_file": filename,
                    "embedding_length": len(embedding_vector),
                    "embedding": embedding_vector
                }, ef, indent=2, ensure_ascii=False)

            log(f"✅ Created embedding: {embed_filename}")

        except Exception as e:
            log(f"❌ Error generating embedding for {filename}: {e}")

log("===== All embeddings generated successfully =====")
