import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import List, Optional

# -------------------------------
# 1️⃣ Configuration & Setup
# -------------------------------

app = FastAPI(title="Lost & Found Unified Embeddings API")

# File path to store the FAISS index permanently
INDEX_FILE = "lost_found_unified.faiss"

# CLIP Model Name
MODEL_NAME = "ViT-B/32"

# Global variables for model and index
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
preprocess = None
faiss_index = None

# Dimension of CLIP ViT-B/32 embeddings
EMBEDDING_DIM = 512

# -------------------------------
# 2️⃣ Lifecycle Events (Startup)
# -------------------------------

@app.on_event("startup")
async def startup_event():
    """
    1. Load the CLIP model into memory.
    2. Load the FAISS index from disk if it exists, otherwise create a new one.
    """
    global model, preprocess, faiss_index

    print(f"🔄 Loading CLIP model ({MODEL_NAME}) on {device}...")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    print("✅ CLIP model loaded.")

    if os.path.exists(INDEX_FILE):
        print(f"📂 Found existing index file at {INDEX_FILE}. Loading...")
        faiss_index = faiss.read_index(INDEX_FILE)
        print(f"✅ Index loaded. Contains {faiss_index.ntotal} vectors.")
    else:
        print("🆕 No existing index found. Creating new FAISS index.")
        # We use IDMap to map vectors to specific item_ids (e.g., database primary keys)
        # We use FlatIP (Inner Product) because normalized vectors + IP = Cosine Similarity
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
        print("✅ New index created.")

# -------------------------------
# 3️⃣ Helper Functions (The Brains)
# -------------------------------

def save_index_to_disk():
    """Saves the current FAISS index to the local file system."""
    if faiss_index:
        faiss.write_index(faiss_index, INDEX_FILE)
        print(f"💾 Index saved to {INDEX_FILE}")

def get_text_embedding(text: str) -> np.ndarray:
    """
    Generates a normalized CLIP embedding for text.
    Shape: (1, 512)
    """
    # Truncate text to fit CLIP's context length (77 tokens)
    text_tokenized = clip.tokenize([text[:77]], truncate=True).to(device)
    
    with torch.no_grad():
        embedding = model.encode_text(text_tokenized)
        embedding /= embedding.norm(dim=-1, keepdim=True) # Normalize for cosine similarity
        
    return embedding.cpu().numpy().astype('float32')

def get_image_embedding(image_file: UploadFile) -> np.ndarray:
    """
    Generates a normalized CLIP embedding for an image.
    Shape: (1, 512)
    """
    # Ensure we are at the start of the file
    image_file.file.seek(0)
    image = Image.open(image_file.file).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True) # Normalize for cosine similarity
        
    return embedding.cpu().numpy().astype('float32')

# -------------------------------
# 4️⃣ API Endpoints
# -------------------------------

@app.post("/upload_item/")
async def upload_item(
    item_id: int,
    background_tasks: BackgroundTasks,
    description: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Upload an item to the vector database.
    - If BOTH Image and Text are provided, we create a Hybrid Embedding (0.5 weight).
    - If only IMAGE is provided, we vectorise the image.
    - If only TEXT is provided, we vectorise the text.
    - Result is stored in the SAME index to allow cross-modal search.
    """
    if not image and not description:
        raise HTTPException(status_code=400, detail="Must provide either an image or a description.")

    image_vec = None
    text_vec = None
    final_vector = None
    modality_used = ""

    # 1. Compute individual embeddings
    if image:
        image_vec = get_image_embedding(image)
    if description:
        text_vec = get_text_embedding(description)

    # 2. Combine Logic (Alpha = 0.5)
    alpha = 0.5

    if image_vec is not None and text_vec is not None:
        # Hybrid Case: Weighted Average
        combined = (alpha * image_vec) + ((1 - alpha) * text_vec)
        
        # IMPORTANT: Re-normalize. 
        # Averaging two unit vectors results in a vector with length < 1.
        # We must normalize back to unit length for Cosine Similarity to work in FAISS.
        norm = np.linalg.norm(combined, axis=1, keepdims=True)
        final_vector = combined / norm
        modality_used = "hybrid"

    elif image_vec is not None:
        final_vector = image_vec
        modality_used = "image"
        
    elif text_vec is not None:
        final_vector = text_vec
        modality_used = "text"

    # 3. Add to FAISS Index
    if final_vector is not None:
        id_array = np.array([item_id], dtype='int64')
        faiss_index.add_with_ids(final_vector, id_array)

    # Schedule a save to disk so we don't block the user response
    background_tasks.add_task(save_index_to_disk)

    return {
        "status": "success",
        "item_id": item_id,
        "indexed_modality": modality_used,
        "total_items_in_index": faiss_index.ntotal
    }

@app.post("/search/")
async def search_item(
    query_text: Optional[str] = Form(None),
    query_image: Optional[UploadFile] = File(None),
    top_k: int = 5
):
    """
    Multimodal Search:
    1. Text Only: Matches semantic meaning.
    2. Image Only: Matches visual similarity.
    3. Hybrid (Text + Image): Weighted average (Alpha 0.5) of both vectors.
    """
    if not query_text and not query_image:
        raise HTTPException(status_code=400, detail="Must provide query_text or query_image.")

    image_vec = None
    text_vec = None
    final_vector = None

    # 1. Compute individual embeddings if inputs exist
    if query_image:
        image_vec = get_image_embedding(query_image)
    
    if query_text:
        text_vec = get_text_embedding(query_text)

    # 2. Combine Logic (Alpha = 0.5)
    alpha = 0.5

    if image_vec is not None and text_vec is not None:
        # Hybrid Case
        combined = (alpha * image_vec) + ((1 - alpha) * text_vec)
        norm = np.linalg.norm(combined, axis=1, keepdims=True)
        final_vector = combined / norm

    elif image_vec is not None:
        final_vector = image_vec
    elif text_vec is not None:
        final_vector = text_vec

    # 3. Perform Search
    # D = Distances (Similarity scores), I = Indices (Item IDs)
    D, I = faiss_index.search(final_vector, top_k)

    # 4. Format results
    results = []
    for score, found_id in zip(D[0], I[0]):
        if found_id == -1: continue # FAISS returns -1 if not enough items found
        results.append({
            "item_id": int(found_id),
            "similarity_score": float(score)
        })

    return {"results": results}

# -------------------------------
# 5️⃣ Manual Management (Optional)
# -------------------------------

@app.get("/force_save/")
def force_save():
    """Manually trigger a save to disk."""
    save_index_to_disk()
    return {"status": "Index saved manually."}

@app.delete("/wipe_index/")
def wipe_index():
    """
    ⚠️ DANGER ZONE ⚠️
    Clears all vectors from memory AND deletes the index file from disk.
    Everything starts from scratch after this.
    """
    global faiss_index
    
    # 1. Reset Memory
    faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
    
    # 2. Delete Disk File
    if os.path.exists(INDEX_FILE):
        try:
            os.remove(INDEX_FILE)
            file_status = "Deleted"
        except Exception as e:
            file_status = f"Error deleting file: {e}"
    else:
        file_status = "Not found"
        
    return {
        "status": "success",
        "message": "Index wiped from memory and disk.",
        "file_status": file_status,
        "current_total_vectors": faiss_index.ntotal
    }

@app.get("/stats/")
def get_stats():
    """Check how many items are in the system."""
    return {
        "total_vectors": faiss_index.ntotal,
        "index_file_size_bytes": os.path.getsize(INDEX_FILE) if os.path.exists(INDEX_FILE) else 0
    }
