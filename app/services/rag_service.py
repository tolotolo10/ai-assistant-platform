# app/services/rag_service.py
from __future__ import annotations
import os
import time
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from app.config import settings
from app.services.ingestion import load_files, load_web, make_chunks

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

from pinecone import Pinecone, ServerlessSpec

# Optional: Chroma
try:
    from langchain_community.vectorstores import Chroma
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False

# Optional: Reranker
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CohereRerank
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


class QueryType(Enum):
    """Types of queries for different handling strategies."""
    METADATA = "metadata"  # Author, title, etc.
    CONTENT = "content"    # Regular content questions
    SUMMARY = "summary"    # What is this about?
    SPECIFIC = "specific"  # Conclusion, results, etc.


class QueryEnhancer:
    """Enhance queries for better retrieval."""
    
    @staticmethod
    def classify_query(query: str) -> QueryType:
        """Classify the type of query."""
        query_lower = query.lower()
        
        # Metadata queries
        if any(word in query_lower for word in [
            'author', 'wrote', 'written', 'who wrote', 'who is the author',
            'title', 'name of', 'called'
        ]):
            return QueryType.METADATA
        
        # Summary queries
        if any(phrase in query_lower for phrase in [
            'what is this about', 'summarize', 'summary', 'overview',
            'what does', 'main topic', 'describe the document'
        ]):
            return QueryType.SUMMARY
        
        # Specific section queries
        if any(word in query_lower for word in [
            'conclusion', 'result', 'finding', 'contribution',
            'limitation', 'future work', 'methodology', 'approach'
        ]):
            return QueryType.SPECIFIC
        
        return QueryType.CONTENT
    
    @staticmethod
    def enhance_query(query: str, query_type: QueryType) -> List[str]:
        """Generate multiple enhanced queries for better retrieval."""
        enhanced = [query]  # Always include original
        
        if query_type == QueryType.METADATA:
            if 'author' in query.lower():
                enhanced.extend([
                    "authors names written by who wrote this paper",
                    "author affiliation university institute",
                    query.replace('?', '') + " names of authors"
                ])
            elif 'title' in query.lower():
                enhanced.extend([
                    "document title name paper called",
                    "title of this research paper article"
                ])
        
        elif query_type == QueryType.SPECIFIC:
            if 'conclusion' in query.lower():
                enhanced.extend([
                    "conclusion section final thoughts summary findings",
                    "concluding remarks future work limitations",
                    "in conclusion we conclude this paper concludes"
                ])
            elif 'result' in query.lower():
                enhanced.extend([
                    "results experiments evaluation performance metrics",
                    "experimental results findings outcomes",
                    "our results show demonstrate indicate"
                ])
        
        elif query_type == QueryType.SUMMARY:
            enhanced.extend([
                "abstract introduction overview summary",
                "this paper presents proposes investigates",
                "main contribution key findings"
            ])
        
        return enhanced


class RAGService:
    def __init__(self) -> None:
        # CRITICAL: Set these attributes FIRST before calling _validate_settings()
        self.vector_db = getattr(settings, 'vector_db', 'pinecone').lower().strip()
        self.index_name = getattr(settings, 'pinecone_index', None)
        self.namespace = getattr(settings, 'pinecone_namespace', 'default').strip()
        
        # NOW we can validate (because self.vector_db exists)
        self._validate_settings()
        
        # Initialize OpenAI clients
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=EMBED_MODEL,
        )
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=getattr(settings, 'default_model', 'gpt-3.5-turbo'),
            temperature=getattr(settings, 'temperature', 0.0),
        )
        
        # Query enhancer
        self.query_enhancer = QueryEnhancer()
        
        # Check embedding dimensions
        dim_probe = len(self.embeddings.embed_query("test"))
        if dim_probe != EMBED_DIM:
            raise RuntimeError(f"Embedding dimension mismatch: {dim_probe} != {EMBED_DIM}")
        
        self._retriever = None
        self._retriever_k = None
        self._pc = None
        
        if self.vector_db == "pinecone":
            self._ensure_pinecone_index()
        elif self.vector_db == "chroma":
            if not HAS_CHROMA:
                raise RuntimeError("Chroma not installed")
        else:
            raise ValueError(f"Unknown vector DB: {self.vector_db}")
        
        logger.info(f"RAGService initialized: {self.vector_db}")
    
    def _validate_settings(self) -> None:
        """Validate required settings."""
        missing = []
        
        # Always required
        if not getattr(settings, 'openai_api_key', None):
            missing.append("OPENAI_API_KEY")
        
        # Get vector_db, defaulting to 'pinecone' if not set
        vector_db = getattr(self, 'vector_db', 'pinecone')
        
        # Check Pinecone settings if using Pinecone
        if vector_db == "pinecone":
            if not getattr(settings, 'pinecone_api_key', None):
                missing.append("PINECONE_API_KEY")
            if not getattr(settings, 'pinecone_index', None):
                missing.append("PINECONE_INDEX")
            if not getattr(settings, 'pinecone_cloud', None):
                missing.append("PINECONE_CLOUD")
            if not getattr(settings, 'pinecone_region', None):
                missing.append("PINECONE_REGION")
        
        if missing:
            raise RuntimeError(f"Missing required settings: {', '.join(missing)}")
    
    def _ensure_pinecone_index(self) -> None:
        """Ensure Pinecone index exists."""
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self._pc = pc
        
        existing = pc.list_indexes()
        names = []
        for ix in existing:
            name = getattr(ix, "name", None) or (ix.get("name") if isinstance(ix, dict) else None)
            if name:
                names.append(name)
        
        if self.index_name not in names:
            logger.info(f"Creating index: {self.index_name}")
            spec = ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            )
            pc.create_index(
                name=self.index_name,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=spec,
            )
            
            # Wait for readiness
            for _ in range(60):
                try:
                    desc = pc.describe_index(self.index_name)
                    status = getattr(desc, "status", {})
                    ready = getattr(status, "ready", None) or (isinstance(status, dict) and status.get("ready"))
                    if ready:
                        break
                except Exception:
                    pass
                time.sleep(0.5)
        
        logger.info(f"Index ready: {self.index_name}")
    
    def ingest(self, file_paths: Optional[List[str]] = None, urls: Optional[List[str]] = None) -> int:
        """Ingest documents with smart chunking."""
        docs = []
        if file_paths:
            docs.extend(load_files(file_paths))
        if urls:
            docs.extend(load_web(urls))
        
        if not docs:
            logger.warning("No documents to ingest")
            return 0
        
        # Create chunks with metadata preservation
        chunks = make_chunks(
            docs,
            chunk_size=512,
            chunk_overlap=128,
            use_semantic=False,  # Set to True if you have semantic chunker
            preserve_metadata_chunks=True
        )
        
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            logger.warning("No chunks created")
            return 0
        
        # Log metadata chunks for debugging
        metadata_chunks = [c for c in chunks if c.metadata.get('type', '').endswith('_metadata')]
        logger.info(f"Metadata chunks: {len(metadata_chunks)}")
        for mc in metadata_chunks:
            logger.info(f"  - {mc.metadata.get('type')}: {mc.page_content[:100]}...")
        
        # Store in vector DB
        if self.vector_db == "pinecone":
            PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace=self.namespace,
                pinecone_api_key=settings.pinecone_api_key,
            )
        else:
            persist_dir = os.path.join(os.getcwd(), ".chroma")
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_dir,
                collection_name=getattr(settings, "chroma_collection", "kb"),
            )
        
        # Reset retriever
        self._retriever = None
        self._retriever_k = None
        
        logger.info(f"Ingested {len(chunks)} chunks")
        return len(chunks)
    
    def _get_retriever(self, k: int = 4) -> BaseRetriever:
        """Get or create retriever."""
        if self._retriever and self._retriever_k == k:
            return self._retriever
        
        if self.vector_db == "pinecone":
            store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=self.namespace,
                pinecone_api_key=settings.pinecone_api_key,
            )
        else:
            persist_dir = os.path.join(os.getcwd(), ".chroma")
            store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
                collection_name=getattr(settings, "chroma_collection", "kb"),
            )
        
        self._retriever = store.as_retriever(search_kwargs={"k": k})
        self._retriever_k = k
        return self._retriever
    
    def _retrieve_with_enhancement(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve documents with query enhancement."""
        # Classify query
        query_type = self.query_enhancer.classify_query(query)
        logger.info(f"Query type: {query_type.value}")
        
        # Enhance query
        enhanced_queries = self.query_enhancer.enhance_query(query, query_type)
        
        # Adjust k based on query type
        if query_type in [QueryType.METADATA, QueryType.SPECIFIC]:
            k = min(k * 2, 12)  # Retrieve more for metadata/specific queries
        
        retriever = self._get_retriever(k=k)
        
        # Retrieve documents for all enhanced queries
        all_docs = []
        seen_contents = set()
        
        for eq in enhanced_queries[:3]:  # Use top 3 enhanced queries
            try:
                docs = retriever.invoke(eq) or []
            except Exception:
                try:
                    docs = retriever.get_relevant_documents(eq) or []
                except Exception:
                    docs = []
            
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        
        # Prioritize metadata chunks for metadata queries
        if query_type == QueryType.METADATA:
            metadata_docs = [d for d in all_docs if d.metadata.get('type', '').endswith('_metadata')]
            regular_docs = [d for d in all_docs if not d.metadata.get('type', '').endswith('_metadata')]
            all_docs = metadata_docs + regular_docs
        
        return all_docs[:k]
    
    def answer(self, question: str, top_k: int = 4) -> dict:
        """Generate answer with enhanced retrieval and prompting."""
        k = max(4, min(top_k, 12))
        
        # Retrieve with enhancement
        docs = self._retrieve_with_enhancement(question, k)
        
        # Filter empty docs
        docs = [d for d in docs if d.page_content.strip()]
        
        if not docs:
            return {"answer": "I don't have information to answer this question.", "sources": []}
        
        # Format context
        def format_doc(doc: Document) -> str:
            meta = doc.metadata or {}
            doc_type = meta.get('type', '')
            source = meta.get('source', 'document')
            
            # Special formatting for metadata chunks
            if doc_type == 'authors_metadata':
                return f"[AUTHOR INFORMATION]\n{doc.page_content}\n"
            elif doc_type == 'title_abstract_metadata':
                return f"[TITLE AND ABSTRACT]\n{doc.page_content}\n"
            elif doc_type == 'conclusion_metadata':
                return f"[CONCLUSION SECTION]\n{doc.page_content}\n"
            elif doc_type == 'results_metadata':
                return f"[RESULTS SECTION]\n{doc.page_content}\n"
            else:
                return f"[Source: {source}]\n{doc.page_content[:1500]}\n"
        
        context = "\n---\n".join(format_doc(d) for d in docs[:k])
        
        # Detect query type for specialized prompting
        query_type = self.query_enhancer.classify_query(question)
        
        # Create appropriate prompt
        if query_type == QueryType.METADATA:
            prompt_template = """You are a helpful assistant. Answer the question using ONLY the provided context.

Special Instructions:
- For author questions: Look for sections marked [AUTHOR INFORMATION] or names near the beginning of documents
- For title questions: Look for [TITLE AND ABSTRACT] sections
- Be precise with names and titles
- If the information is not in the context, say "I couldn't find this information in the document"

Question: {question}

Context:
{context}

Answer:"""
        
        elif query_type == QueryType.SPECIFIC:
            prompt_template = """You are a helpful assistant. Answer using ONLY the provided context.

Special Instructions:
- Look for sections marked with [CONCLUSION SECTION] or [RESULTS SECTION] if relevant
- Provide specific details from the document
- Quote relevant passages when appropriate

Question: {question}

Context:
{context}

Answer:"""
        
        else:
            prompt_template = """You are a helpful assistant. Answer using ONLY the provided context.
Do not use outside knowledge. If you cannot answer from the context, say so.

Question: {question}

Context:
{context}

Answer:"""
        
        # Generate answer
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "context"]
        )
        
        llm_input = prompt.format(question=question, context=context)
        
        try:
            response = self.llm.invoke(llm_input)
            answer_text = getattr(response, "content", str(response)).strip()
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer_text = "Error generating answer"
        
        # Post-process for common issues
        if "I don't know" in answer_text or not answer_text:
            # Try to extract from metadata if available
            for doc in docs:
                meta = doc.metadata
                if query_type == QueryType.METADATA:
                    if 'author' in question.lower() and meta.get('authors'):
                        answer_text = f"The authors are: {', '.join(meta['authors'])}"
                        break
                    elif 'title' in question.lower() and meta.get('title'):
                        answer_text = f"The title is: {meta['title']}"
                        break
        
        # Prepare sources
        sources = []
        seen_sources = set()
        for doc in docs[:min(k, 5)]:
            meta = doc.metadata or {}
            source = meta.get('source', 'document')
            if source not in seen_sources:
                sources.append({
                    'metadata': meta,
                    'snippet': doc.page_content[:300]
                })
                seen_sources.add(source)
        
        return {
            "answer": answer_text,
            "sources": sources
        }
    
    def diagnose(self) -> Dict[str, Any]:
        """Diagnostic information."""
        return {
            "vector_db": self.vector_db,
            "index": self.index_name,
            "namespace": self.namespace,
            "embed_model": EMBED_MODEL,
            "embed_dim": EMBED_DIM,
            "has_reranker": HAS_RERANKER,
        }