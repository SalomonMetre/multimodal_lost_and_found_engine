# 🔍 Lost & Found Unified Embeddings API

> **A Multimodal Search Engine that bridges the gap between Visual Appearance and Natural Language.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![OpenAI CLIP](https://img.shields.io/badge/AI-CLIP-orange)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-red)

## 📖 Overview

The "Lost & Found Problem" is fundamentally a communication gap:
* **The Loser** describes an item in text ("Lost a vintage brown bag").
* **The Finder** takes a picture of the item (Visual data).

Traditional databases cannot match these two. This API uses **OpenAI's CLIP** model to project both images and text into a shared 512-dimensional vector space. It implements an **"Early Fusion" architecture**, allowing for hybrid storage and search where visual cues and semantic descriptions are mathematically combined.

## 🚀 Key Features

* **Multimodal Indexing:** Upload an Image, a Text Description, or **Both**.
* **Hybrid Embeddings:** If both image and text are provided, they are fused into a single weighted vector (Semantic Centroid) before indexing.
* **Cross-Modal Search:**
    * Search with Text → Find Images.
    * Search with Image → Find Text Descriptions.
    * Search with Hybrid → Find the perfect match.
* **High Performance:** Uses **FAISS (Facebook AI Similarity Search)** for millisecond-latency vector retrieval.
* **Persistence:** Automatically saves and loads the vector index to disk.

---

## ⚙️ Architecture: The "Early Fusion" Approach

Unlike traditional systems that maintain separate indexes for images and text (Late Fusion), this engine combines them **before** storage.

$$\vec{V}_{final} = \text{Normalize}( \alpha \cdot \vec{I}_{image} + (1-\alpha) \cdot \vec{T}_{text} )$$

* **$\alpha$ (Alpha):** Currently set to `0.5`, giving equal weight to visual appearance and semantic meaning.
* **Normalization:** Ensures the resulting hybrid vector remains on the unit hypersphere, preserving Cosine Similarity properties for FAISS.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/lost-found-api.git](https://github.com/yourusername/lost-found-api.git)
cd lost-found-api
````

### 2\. Install Dependencies

You need PyTorch, CLIP, FAISS, and FastAPI.

```bash
# Install PyTorch (Visit pytorch.org for your specific CUDA version if needed)
pip install torch torchvision

# Install OpenAI CLIP
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)

# Install FAISS (CPU version usually sufficient for small/medium datasets)
pip install faiss-cpu

# Install API framework
pip install fastapi uvicorn python-multipart pillow numpy
```

### 3\. Run the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

-----

## 📡 API Endpoints

### 1\. Upload Item (`POST /upload_item/`)

Adds a lost or found item to the vector index.

  * **Input:** `item_id` (int), `image` (file, optional), `description` (text, optional).
  * **Logic:**
      * If **Both** are provided: Creates a Hybrid Vector (Visuals + Semantics).
      * If **Image Only**: Creates a Visual Vector.
      * If **Text Only**: Creates a Semantic Text Vector.
  * **Returns:** Status and indexed modality.

### 2\. Search (`POST /search/`)

Finds the most similar items in the database.

  * **Input:** `query_text` (text, optional), `query_image` (file, optional), `top_k` (int).
  * **Logic:** Converts inputs into a query vector (handling hybrid inputs identically to uploads) and scans the FAISS index for nearest neighbors.
  * **Returns:** List of `item_id`s and their `similarity_score` (0.0 to 1.0).

### 3\. Get Stats (`GET /stats/`)

Returns the health of the system.

  * **Returns:** Total number of vectors in memory and the size of the index file on disk.

### 4\. Force Save (`GET /force_save/`)

Manually triggers a flush of the in-memory index to the disk (`lost_found_unified.faiss`).

  * *Note: The system also saves automatically in the background after uploads.*

### 5\. Wipe Index (`DELETE /wipe_index/`)

⚠️ **Destructive Action.**

  * Clears all vectors from memory.
  * Deletes the persistent `.faiss` file from the disk.
  * Resets the system to a blank slate.

-----

## 🧪 Usage Examples (cURL)

**1. Upload a Found Item (Image only)**

```bash
curl -X POST "[http://127.0.0.1:8000/upload_item/](http://127.0.0.1:8000/upload_item/)" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "item_id=101" \
     -F "image=@/path/to/photo_of_keys.jpg"
```

**2. Upload a Lost Item (Text + Image)**

```bash
curl -X POST "[http://127.0.0.1:8000/upload_item/](http://127.0.0.1:8000/upload_item/)" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "item_id=202" \
     -F "description=Blue umbrella with a wooden handle" \
     -F "image=@/path/to/stock_umbrella.jpg"
```

**3. Search for an Item**

```bash
curl -X POST "[http://127.0.0.1:8000/search/?top_k=3](http://127.0.0.1:8000/search/?top_k=3)" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "query_text=Blue umbrella"
```

-----

## 🧠 Theory: Interpreting Scores

Scores returned are **Cosine Similarity**.

  * **1.0:** Perfect duplicate.
  * **\> 0.85:** High visual match (likely same object).
  * **0.30 - 0.60:** Good semantic/conceptual match (e.g., Text "Phone" matches Image of Phone).
  * **\< 0.20:** Likely noise/irrelevant.

## 📝 License

MIT
