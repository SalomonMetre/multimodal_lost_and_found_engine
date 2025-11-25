# 🔍 Lost & Found Unified Embeddings API

A Multimodal Search Engine that bridges visual appearance and natural language using CLIP + Early Fusion + FAISS.

## 📖 Overview

Classic Lost & Found systems fail because one side has only text and the other side has only a photo.  
This project eliminates the modality gap by embedding both images and text into the same 512-dimensional CLIP space.  
When both modalities are available, an Early Fusion step combines them into a single superior vector.

Capabilities:
- Text → Image retrieval
- Image → Text retrieval
- Hybrid query (text + image) → highest precision
- Millisecond-scale search via FAISS

## 🚀 Key Features

- Single unified FAISS index (no separate text/image indexes)
- Early Fusion (α-weighted sum + re-normalization) for hybrid items and queries
- Persistent index with automatic save/load
- Pure HTTP + multipart/form-data API (no Pydantic forms, no extra wrappers
- Fully managed with uv (pyproject.toml + uv.lock already committed)

## ⚙️ Early Fusion Formula

$$
\vec{v}_{\text{final}} = \text{Normalize}\left(\alpha \cdot \vec{v}_{\text{image}} + (1-\alpha) \cdot \vec{v}_{\text{text}}\right)
$$

α = 0.5 by default → stays on the CLIP unit hypersphere → perfect cosine similarity with FAISS.

## 🛠️ Installation & Run (uv)

Your repository already contains `pyproject.toml` and `uv.lock`, so everything is ready:

```bash
# 1. Clone and enter
git clone https://github.com/SalomonMetre/lost_and_found_engine.git
cd lost_and_found_engine

# 2. Install dependencies + create venv in one step
uv sync --frozen          # respects uv.lock, uses uv.lock

# 3. Run the server
uv run uvicorn main:app --reload
```

Server will be available at http://127.0.0.1:8000

## 📡 API Endpoints

| Method | Endpoint                  | Purpose                                      |
|--------|---------------------------|----------------------------------------------|
| POST   | `/upload_item/`           | Index an item (image, text or both)          |
| POST   | `/search/?top_k=N`        | Search with text, image or both              |
| GET    | `/stats/`                 | Index statistics                             |
| GET    | `/force_save/`            | Force persistence to disk                    |
| DELETE | `/wipe_index/`            | ⚠️ Delete everything                         |

All endpoints use standard `multipart/form-data` (regular HTML file uploads).

### Upload examples (cURL)

```bash
# Image only
curl -X POST http://127.0.0.1:8000/upload_item/ \
  -F "item_id=101" \
  -F "image=@./keys.jpg"

# Text + Image (Early Fusion)
curl -X POST http://127.0.0.1:8000/upload_item/ \
  -F "item_id=202" \
  -F "description=Red backpack with white stripes" \
  -F "image=@./backpack.jpg"
```

### Search examples (cURL)

```bash
# Text query
curl -X POST "http://127.0.0.1:8000/search/?top_k=5" \
  -F "query_text=Red backpack"

# Hybrid query – best accuracy
curl -X POST "http://127.0.0.1:8000/search/?top_k=5" \
  -F "query_text=black leather wallet" \
  -F "query_image=@./found_wallet.jpg"
```

## 🧠 Arbitrary Similarity Interpretation

| Score   | Meaning                       |
|---------|-------------------------------|
| 1.00    | Exact duplicate               |
| ≥ 0.85  | Very strong match             |
| 0.70–0.85 | Clear match                 |
| 0.50–0.70 | Reasonable / partial match  |
| < 0.50  | Weak or unrelated             |

## 📂 Project Structure

```
lost_and_found_engine/
├── .gitignore
├── .python-version
├── pyproject.toml          # uv metadata
├── uv.lock                 # exact dependency lockfile
├── main.py                 # FastAPI app
├── clip_utils.py
├── faiss_index.py
├── hybrid_fusion.py
├── lost_found_unified.faiss   # auto-created/saved
└── README.md
```

## 📝 License

MIT License
