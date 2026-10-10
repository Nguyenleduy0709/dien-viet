# Diện Việt — AI-Powered Vietnamese Traditional Fashion Discovery

Diện Việt is an AI-powered prototype designed to help users discover Vietnamese traditional clothing, especially áo dài, based on their preferred occasion, color, style, and personal preferences.

The project combines semantic search, text embeddings, clothing metadata, and cultural knowledge to make traditional fashion discovery more accessible and intuitive.

## Overview

Finding a suitable traditional Vietnamese outfit can be challenging when users need to consider different occasions, styles, colors, and cultural contexts.

Diện Việt explores how AI can help users search for relevant clothing through natural-language queries rather than relying entirely on manual browsing or exact keyword matching.

The current prototype focuses on an áo dài catalog and a local semantic retrieval pipeline.

## Key Features

- **Semantic Search:** Retrieves clothing items based on the semantic meaning of user queries and catalog descriptions.
- **Attribute-Aware Ranking:** Uses available metadata, such as clothing category, color, style, and occasion, to improve search relevance.
- **AI-Assisted Image Annotation:** Uses Gemini Vision in an offline workflow to generate candidate descriptions and labels for clothing images.
- **Embedding-Based Retrieval:** Converts text descriptions into vector embeddings and stores them for local similarity search.
- **Cultural Knowledge Base:** Organizes cultural notes and draft styling rules in JSON files.
- **FastAPI Backend:** Provides the backend API for search and application functionality.
- **Data Processing Pipeline:** Includes scripts for downloading images, annotating the catalog, building embeddings, and testing search results.

## How It Works

The system follows an offline preparation pipeline and an online search workflow.

### Offline Preparation

1. Collect áo dài images for the catalog.
2. Use Gemini Vision to generate candidate descriptions and metadata.
3. Store annotated records in JSON format.
4. Generate text embeddings from catalog descriptions.
5. Save the embedding matrix and corresponding catalog IDs for retrieval.

### Online Search

1. The user enters a natural-language query.
2. The backend processes the query using the search pipeline.
3. Semantic similarity and available clothing attributes help rank candidate items.
4. The system returns relevant clothing results from the catalog.

```text
Offline Pipeline
Images → AI Annotation → Catalog Metadata
       → Text Embeddings → Local Vector Index

Online Pipeline
User Query → Backend Search
           → Semantic Retrieval
           → Attribute-Aware Ranking
           → Recommended Clothing
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend logic and data processing |
| FastAPI | Backend API framework |
| Sentence Transformers | Text embedding generation |
| NumPy | Embedding storage and numerical operations |
| JSON | Catalog metadata and cultural knowledge |
| Gemini Vision API | AI-assisted image annotation |
| HTML, CSS, JavaScript | Frontend implementation |

## Project Structure

```text
vietphuc5.5/
├── backend/
│   ├── main.py
│   ├── ai_search_service.py
│   ├── embedding_utils.py
│   ├── cultural.py
│   ├── search_service.py
│   ├── schemas.py
│   └── mock_catalog.json
├── data/
│   ├── catalog/
│   ├── cultural/
│   ├── embeddings/
│   ├── metadata/
│   └── raw/
│       └── ao_dai/
├── frontend/
│   └── index.html
├── scripts/
│   ├── annotate_ao_dai.py
│   ├── build_embeddings.py
│   ├── download_ao_dai.py
│   └── test_search.py
├── .env.example
├── .gitignore
├── fetch_ao_dai.py
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python
- Git
- A Gemini API key if you want to run the AI-assisted annotation workflow

### 1. Clone the Repository

```bash
git clone https://github.com/Nguyenleduy0709/dien-viet.git
cd dien-viet
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

If you need to run the Gemini-based annotation workflow, copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and add your own API key:

```dotenv
GEMINI_API_KEY=your_real_api_key_here
```

**Security note:** Never commit `.env` files, API keys, or other credentials to GitHub. The `.env.example` file should contain placeholders only.

### 5. Run the Backend

From the repository root, run:

```powershell
uvicorn backend.main:app --reload --port 8000
```

Alternatively, if you are using the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

Once the server starts, open:

- **Backend:** http://127.0.0.1:8000
- **Interactive API Documentation:** http://127.0.0.1:8000/docs

## Data and Embedding Preparation

The repository includes scripts for preparing and testing the catalog:

| Script | Purpose |
|---|---|
| `scripts/download_ao_dai.py` | Downloads áo dài image data |
| `scripts/annotate_ao_dai.py` | Generates candidate descriptions and labels |
| `scripts/build_embeddings.py` | Builds the embedding index |
| `scripts/test_search.py` | Tests search functionality |

Review each script's configuration before running it. The annotation workflow may consume API quota when using Gemini.

## Current Limitations

- This project is a prototype and is still under development.
- Search quality depends on catalog coverage, metadata quality, and the embedding configuration.
- Some images may have incomplete metadata or annotation errors.
- Cultural notes and styling rules require further validation against reliable sources.
- Image licensing and usage permissions should be checked before redistributing the dataset or using it in production.
- The current repository focuses on local development rather than production deployment.

## Future Improvements

- Improve the quality and coverage of image annotations.
- Evaluate retrieval performance using a documented set of test queries.
- Refine the balance between semantic similarity and attribute matching.
- Expand and validate the cultural knowledge base.
- Improve the frontend and backend integration.
- Add more comprehensive testing and deployment instructions.

## Acknowledgements

The prototype uses an áo dài image collection associated with the [`Dangindev/viet-cultural-vqa`](https://huggingface.co/datasets/Dangindev/viet-cultural-vqa) dataset.

Please review the dataset's original license and terms before redistributing images or using them beyond the permitted scope.

## License

No project license has been specified yet. Unless a license is added, permissions to reuse or redistribute this project's code should not be assumed. Dataset and image licenses may have separate terms.
