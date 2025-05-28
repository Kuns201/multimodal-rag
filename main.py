import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
import fitz
import openai
import faiss
import numpy as np
import os
import json
from PIL import Image
from io import BytesIO
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas import evaluate
from datasets import Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import CrossEncoder
import torch
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import shutil

app = FastAPI()

UPLOAD_DIR = "uploads"
INDEX_DIR = "index"
BM25_DIR = "bm25_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(BM25_DIR, exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def embed_text(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def expand_query(question):
    prompt = f"Expand this query to include synonyms and related terms: {question}"
    completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message['content']

def blip_caption(image_bytes):
    raw_image = Image.open(BytesIO(image_bytes)).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Form(...), onedrive_url: str = Form(...), role: str = Form(...)):
    filepath = os.path.join(UPLOAD_DIR, f"{user_id}_{file.filename}")
    with open(filepath, "wb") as f:
        f.write(await file.read())

    doc = fitz.open(filepath)
    texts, metadata = [], []
    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    bm25_path = os.path.join(BM25_DIR, user_id)
    if os.path.exists(bm25_path): shutil.rmtree(bm25_path)
    os.makedirs(bm25_path)
    ix = create_in(bm25_path, schema)
    writer = ix.writer()
    table_counter, image_counter = 1, 1

    for page_num, page in enumerate(doc):
        text = page.get_text()
        page_image = page.get_pixmap()
        image_bytes = page_image.tobytes("png")

        if text.strip():
            tag = f"table_{table_counter}"
            table_counter += 1
            texts.append(text)
            metadata.append({
                "filename": file.filename, "page": page_num + 1, "text": text, "type": "text", "tag": tag,
                "url": f"{onedrive_url}&page={page_num + 1}", "role": role
            })
            writer.add_document(id=tag, content=text)

        caption = blip_caption(image_bytes)
        tag = f"image_{image_counter}"
        image_counter += 1
        texts.append(caption)
        metadata.append({
            "filename": file.filename, "page": page_num + 1, "text": caption, "type": "image", "tag": tag,
            "url": f"{onedrive_url}&page={page_num + 1}", "role": role
        })
        writer.add_document(id=tag, content=caption)

    writer.commit()

    embeddings = [embed_text(t) for t in texts]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, os.path.join(INDEX_DIR, f"{user_id}.index"))
    with open(os.path.join(INDEX_DIR, f"{user_id}_meta.json"), "w") as f:
        json.dump(metadata, f)

    return {"status": "indexed", "pages": len(doc)}

@app.post("/query")
async def query(user_id: str = Form(...), question: str = Form(...), role: str = Form(default=None), expand: bool = Form(default=False)):
    if expand:
        question = expand_query(question)

    ix = open_dir(os.path.join(BM25_DIR, user_id))
    parser = QueryParser("content", ix.schema)
    q = parser.parse(question)
    with ix.searcher() as searcher:
        bm25_results = searcher.search(q, limit=10)
        bm25_ids = [r["id"] for r in bm25_results]

    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, f"{user_id}.index"))
    with open(os.path.join(INDEX_DIR, f"{user_id}_meta.json")) as f:
        meta = json.load(f)

    q_embed = embed_text(question)
    D, I = faiss_index.search(np.array([q_embed]).astype("float32"), k=10)
    faiss_ids = [meta[i]["tag"] for i in I[0] if i < len(meta)]

    # Merge and deduplicate
    hybrid_ids = list(set(bm25_ids + faiss_ids))
    hybrid_chunks = [m for m in meta if m["tag"] in hybrid_ids and (role is None or m["role"] == role)]

    # Rerank top 3
    pairs = [[question, chunk["text"]] for chunk in hybrid_chunks]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(hybrid_chunks, scores), key=lambda x: x[1], reverse=True)[:3]
    top_chunks = [x[0] for x in reranked]

    context = "\n".join([c["text"] for c in top_chunks])
    prompt = f"Use the following sources to answer:\n{context}\n\nQuestion: {question}"

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = completion.choices[0].message["content"]

    ds = Dataset.from_list([{
        "question": question,
        "answer": answer,
        "contexts": [c["text"] for c in top_chunks],
        "ground_truth": answer
    }])
    results = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    

    return {
        "answer": answer,
        "sources": [{"filename": c["filename"], "page": c["page"], "url": c["url"], "tag": c["tag"]} for c in top_chunks],
        "ragas_scores": results
    }

