import os
from app.services.rag_service import RAGService
from pinecone import Pinecone
from app.config import settings

# Find the PDF
pdf_path = None
for root, dirs, files in os.walk('.'):
    for file in files:
        if 'INR' in file and file.endswith('.pdf'):
            pdf_path = os.path.join(root, file)
            break
    if pdf_path:
        break

if not pdf_path:
    print("ERROR: Could not find INR.pdf")
    print("Please put the PDF in the current directory or update the path")
    exit(1)

print(f"Found PDF at: {pdf_path}")

# Optional: Clear old data
clear_old = input("Clear old data from index? (y/n): ").lower() == 'y'
if clear_old:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index)
    index.delete(delete_all=True, namespace=settings.pinecone_namespace or "default")
    print("Cleared old data")

# Ingest with new system
rag = RAGService()
num_chunks = rag.ingest(file_paths=[pdf_path])
print(f"Created {num_chunks} chunks")

# Test queries
test_questions = [
    "Who are the authors of this document?",
    "What is the title?",
]

for q in test_questions:
    result = rag.answer(q, top_k=8)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")