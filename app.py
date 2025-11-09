import os
import io
import time
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

from rag.logger import get_logger, ensure_log_setup, tail_log, clear_log_file
from rag.cache import DiskKVCache
from rag.ingestion import extract_texts_from_pdfs
from rag.chunking import chunk_document_hybrid
from rag.embeddings import EmbeddingModel
from rag.vectorstore import ChromaVectorStore
from rag.retriever import retrieve_relevant_chunks, RetrievalConfig
from rag.llm import LLMClient, build_prompt_with_citations, compute_answer_groundedness
from rag.web_search import web_search_and_summarize
from rag.types import DocumentChunk


# ---------- Constants and Paths ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

ensure_log_setup(LOGS_DIR)
logger = get_logger(__name__)


# ---------- Session State ----------
def init_session_state() -> None:
	"""Initialize Streamlit session state variables."""
	if "history" not in st.session_state:
		st.session_state.history = []  # List[Dict[str, Any]] with question, answer, sources
	if "vector_ready" not in st.session_state:
		st.session_state.vector_ready = False
	if "last_retrieval" not in st.session_state:
		st.session_state.last_retrieval = None
	if "internet_allowed" not in st.session_state:
		st.session_state.internet_allowed = False
	if "web_search_requested" not in st.session_state:
		st.session_state.web_search_requested = False
	if "collection_name" not in st.session_state:
		st.session_state.collection_name = "documents"
	if "model_backend" not in st.session_state:
		st.session_state.model_backend = "gpt4all"  # options: gpt4all, llama.cpp
	if "gpt4all_model" not in st.session_state:
		st.session_state.gpt4all_model = "Mistral-7B-Instruct.Q4_0.gguf"
	if "llama_cpp_model_path" not in st.session_state:
		st.session_state.llama_cpp_model_path = ""
	if "answer_style" not in st.session_state:
		st.session_state.answer_style = "detailed"
	if "target_sentences" not in st.session_state:
		st.session_state.target_sentences = 10


# ---------- App Setup ----------
@st.cache_resource(show_spinner=False)
def get_cache() -> DiskKVCache:
	return DiskKVCache(cache_dir=CACHE_DIR)


@st.cache_resource(show_spinner=False)
def get_embedder() -> EmbeddingModel:
	return EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_vector_store(collection_name: str) -> ChromaVectorStore:
	return ChromaVectorStore(persist_dir=CHROMA_DIR, collection_name=collection_name)


@st.cache_resource(show_spinner=False)
def get_llm(backend: str, gpt4all_model: str, llama_cpp_model_path: str) -> LLMClient:
	return LLMClient(backend=backend, gpt4all_model=gpt4all_model, llama_cpp_model_path=llama_cpp_model_path)


# ---------- UI Components ----------
def sidebar_controls() -> None:
	st.sidebar.header("Settings")
	st.session_state.collection_name = st.sidebar.text_input("Collection name", st.session_state.collection_name)
	st.session_state.internet_allowed = st.sidebar.checkbox("Enable Internet Fallback", value=st.session_state.internet_allowed)

	st.sidebar.subheader("Retrieval")
	top_k = st.sidebar.slider("Top-k chunks", min_value=1, max_value=20, value=5, step=1)
	sim_threshold = st.sidebar.slider("Similarity threshold (0.0-1.0)", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
	st.session_state.retrieval_mode = st.sidebar.selectbox("Retriever mode", options=["hybrid", "dense", "lexical"], index=0)
	st.session_state.rerank_mode = st.sidebar.selectbox("Rerank mode", options=["mmr", "cross-encoder", "none"], index=0)
	st.session_state.mmr_lambda = st.sidebar.slider("MMR lambda (diversity)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
	st.session_state.candidate_k = st.sidebar.slider("Candidate pool size", min_value=top_k, max_value=max(40, top_k * 4), value=max(20, top_k * 3), step=1)

	st.sidebar.subheader("LLM")
	st.session_state.model_backend = st.sidebar.selectbox("Backend", options=["gpt4all", "llama.cpp"], index=0)
	st.session_state.gpt4all_model = st.sidebar.text_input("GPT4All model filename (in %USERPROFILE%/.cache/gpt4all/)", value=st.session_state.gpt4all_model)
	st.session_state.llama_cpp_model_path = st.sidebar.text_input("llama.cpp GGUF path (optional)", value=st.session_state.llama_cpp_model_path)
	st.session_state.answer_style = st.sidebar.selectbox("Answer style", options=["detailed", "concise"], index=0)
	st.session_state.target_sentences = st.sidebar.slider("Target sentences", min_value=4, max_value=14, value=10, step=1)

	st.sidebar.subheader("Maintenance")
	col_a, col_b = st.sidebar.columns(2)
	if col_a.button("Clear Vector DB", use_container_width=True):
		store = get_vector_store(st.session_state.collection_name)
		store.reset()
		st.session_state.vector_ready = False
		st.toast("Vector DB cleared.", icon="🧹")
	if col_b.button("Clear Cache", type="secondary", use_container_width=True):
		get_cache().clear()
		st.toast("Cache cleared.", icon="🧹")
	if st.sidebar.button("Clear Logs", type="secondary", use_container_width=True):
		clear_log_file(LOGS_DIR)
		st.toast("Logs cleared.", icon="🧽")

	st.sidebar.subheader("Monitoring")
	with st.sidebar.expander("View Recent Logs", expanded=False):
		st.code(tail_log(LOGS_DIR, lines=200))

	st.sidebar.caption("Tip: Upload PDFs first, then ask your question.")
	return top_k, sim_threshold


def upload_and_ingest(embedder: EmbeddingModel, store: ChromaVectorStore) -> Tuple[int, int]:
	"""Handle file uploads and ingestion to vector DB. Returns (#docs, #chunks)."""
	uploaded = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
	if not uploaded:
		return 0, 0

	total_chunks = 0
	num_docs = 0
	for file in uploaded:
		save_path = os.path.join(UPLOADS_DIR, file.name)
		with open(save_path, "wb") as f:
			f.write(file.getbuffer())
		logger.info("Saved upload: %s", save_path)

		text_pages = extract_texts_from_pdfs([save_path])
		if not text_pages:
			continue
		for doc in text_pages:
			num_docs += 1
			chunks: List[DocumentChunk] = chunk_document_hybrid(
				document_id=doc["document_id"],
				filename=doc["filename"],
				pages=doc["pages_text"],
				max_tokens=480,
				overlap=40,
				small_chunk_tokens=160,
				sentence_window=4,
				sentence_stride=3,
			)
			if chunks:
				embeddings = embedder.embed_texts([c.text for c in chunks])
				store.add_chunks(chunks, embeddings)
				total_chunks += len(chunks)
				logger.info("Ingested %s: %d chunks", doc["filename"], len(chunks))

	st.session_state.vector_ready = total_chunks > 0
	return num_docs, total_chunks


def ask_question_ui(embedder: EmbeddingModel, store: ChromaVectorStore, llm: LLMClient, top_k: int, sim_threshold: float) -> None:
	st.divider()
	st.subheader("Ask a question")
	query = st.text_input("Enter your question", value="", placeholder="e.g., What are the key findings from the uploaded reports?")
	ask_col1, ask_col2 = st.columns([3, 1])
	with ask_col1:
		submit = st.button("Get Answer", type="primary", use_container_width=True)
	with ask_col2:
		web_btn_label = "Search Internet Now" if st.session_state.internet_allowed else "Ask to Search Internet"
		web_now = st.button(web_btn_label, use_container_width=True)

	if web_now:
		st.session_state.web_search_requested = True
	if not submit:
		return
	if not query.strip():
		st.warning("Please enter a question.")
		return
	if not st.session_state.vector_ready:
		st.warning("No documents indexed yet. Upload PDFs first.")
		return

	with st.spinner("Retrieving relevant context..."):
		retrieval_cfg = RetrievalConfig(
			top_k=top_k,
			similarity_threshold=sim_threshold,
			mode=st.session_state.retrieval_mode,
			rerank=st.session_state.rerank_mode,
			mmr_lambda=float(st.session_state.mmr_lambda),
			candidate_k=int(st.session_state.candidate_k),
		)
		results = retrieve_relevant_chunks(
			query=query,
			embedder=embedder,
			store=store,
			config=retrieval_cfg,
		)
		st.session_state.last_retrieval = results

	# Decide fallback
	use_internet = False
	if not results.chunks or (results.max_score is not None and results.max_score < sim_threshold):
		if st.session_state.internet_allowed or st.session_state.web_search_requested:
			use_internet = True
		else:
			st.info("Not found in PDFs. Do you want me to search the internet?")
			st.session_state.web_search_requested = False

	# Build prompt and answer
	with st.spinner("Generating answer..."):
		history: List[Dict[str, Any]] = st.session_state.history
		answer_style = st.session_state.answer_style
		ts = int(st.session_state.target_sentences)
		# Heuristic: tokens ~ sentences * 90, clipped
		max_tokens = max(384, min(1280, ts * 90))
		if use_internet:
			web_summary, web_sources = web_search_and_summarize(query=query, llm=llm, max_pages=3)
			context_chunks = web_summary
			prompt = build_prompt_with_citations(
				question=query,
				context_blocks=[context_chunks],
				conversation_history=history,
				require_citations=False,
				answer_style=answer_style,
				target_sentences=ts,
			)
			answer = llm.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.3)
			citations = web_sources
			groundedness = None
		else:
			# Deduplicate by (source, page) keeping highest-scoring per pair
			unique_best = {}
			for item in results.chunks:
				key = (item.chunk.source, item.chunk.page)
				if key not in unique_best or item.score > unique_best[key].score:
					unique_best[key] = item
			unique_items = sorted(unique_best.values(), key=lambda x: -x.score)

			# Limit number of context blocks based on target sentences to keep answer focused
			context_limit = min(16, max(4, ts)) if answer_style == "detailed" else min(8, max(3, ts // 2))
			context_blocks = [it.chunk.text for it in unique_items[:context_limit]]
			citations = [
				{"source": it.chunk.source, "page": it.chunk.page, "score": it.score} for it in unique_items[:context_limit]
			]
			prompt = build_prompt_with_citations(
				question=query,
				context_blocks=context_blocks,
				conversation_history=history,
				require_citations=True,
				answer_style=answer_style,
				target_sentences=ts,
			)
			answer = llm.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.3)
			groundedness = compute_answer_groundedness(answer_text=answer, context_blocks=context_blocks, embedder=embedder)

	# Display
	st.markdown("### Final Answer")
	st.write(answer)

	# Top sources summary (PDF name and counts); only for local retrieval
	if not st.session_state.internet_allowed or not st.session_state.web_search_requested:
		if citations:
			source_counts = {}
			for c in citations:
				src = c.get("source") if isinstance(c, dict) else None
				if src:
					source_counts[src] = source_counts.get(src, 0) + 1
			top_sources = sorted(source_counts.items(), key=lambda kv: -kv[1])[:3]
			if top_sources:
				st.caption("Top sources:")
				st.markdown(", ".join(f"{name} ({count})" for name, count in top_sources))

	if groundedness is not None:
		conf_color = "🟢" if groundedness >= 0.7 else ("🟡" if groundedness >= 0.5 else "🔴")
		st.caption(f"{conf_color} Groundedness score: {groundedness:.2f} (compare answer vs. retrieved context)")

	with st.expander("Retrieved Context and Citations", expanded=False):
		if use_internet:
			for i, src in enumerate(citations, start=1):
				st.markdown(f"- [{i}] {src.get('title','')} ({src.get('url','')})")
		else:
			for i, c in enumerate(citations, start=1):
				st.markdown(f"- [{i}] {c['source']} (page {c['page']}) — score={c['score']:.2f}")

	# Append history and cache
	st.session_state.history.append(
		{
			"question": query,
			"answer": answer,
			"citations": citations,
			"used_internet": use_internet,
			"ts": time.time(),
		}
	)
	get_cache().set("last_answer", st.session_state.history[-1])


def monitoring_panel(store: ChromaVectorStore) -> None:
	st.divider()
	st.subheader("Monitoring")
	col1, col2 = st.columns(2)
	with col1:
		st.metric("Vector DB documents", value=store.count())
		st.metric("History length", value=len(st.session_state.history))
	with col2:
		last = get_cache().get("last_answer")
		if last:
			st.caption("Last cached answer:")
			st.json({"question": last["question"], "used_internet": last["used_internet"]})
	with st.expander("Recent Logs", expanded=False):
		st.code(tail_log(LOGS_DIR, lines=300))

def diagnostics_panel(embedder: EmbeddingModel, store: ChromaVectorStore) -> None:
	st.divider()
	st.subheader("Diagnostics")
	with st.expander("PDF Extraction Overview", expanded=False):
		pdfs = [os.path.join(UPLOADS_DIR, f) for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".pdf")]
		if not pdfs:
			st.info("No PDFs found in uploads directory.")
		else:
			st.write(f"Found {len(pdfs)} PDF(s).")
			if st.button("Scan PDFs", use_container_width=False):
				from rag.ingestion import extract_texts_from_pdfs as _scan
				results = _scan(pdfs)
				rows = []
				for r in results:
					pages = r["pages_text"]
					lengths = [len(p or "") for p in pages]
					rows.append(
						{
							"filename": r["filename"],
							"pages": len(pages),
							"avg_chars_per_page": int(sum(lengths) / max(1, len(lengths))),
							"min_chars": min(lengths) if lengths else 0,
							"max_chars": max(lengths) if lengths else 0,
						}
					)
				st.dataframe(rows, use_container_width=True)

	with st.expander("Chunking Preview", expanded=False):
		pdfs = [os.path.join(UPLOADS_DIR, f) for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".pdf")]
		if pdfs:
			fsel = st.selectbox("Select a PDF to preview chunks", options=[os.path.basename(p) for p in pdfs])
			max_tokens = st.number_input("Max tokens", min_value=100, max_value=2000, value=500, step=50)
			overlap = st.number_input("Overlap", min_value=0, max_value=200, value=50, step=10)
			if st.button("Preview chunks", key="preview_chunks"):
				from rag.ingestion import extract_texts_from_pdfs as _scan
				from rag.chunking import chunk_document as _chunk
				path = os.path.join(UPLOADS_DIR, fsel)
				res = _scan([path])
				if res:
					doc = res[0]
					chunks = _chunk(doc["document_id"], doc["filename"], doc["pages_text"], max_tokens=int(max_tokens), overlap=int(overlap))
					st.write(f"Total chunks: {len(chunks)}")
					for c in chunks[:5]:
						st.code((c.text[:600] + ("..." if len(c.text) > 600 else "")))
				else:
					st.warning("Failed to read selected PDF.")
		else:
			st.info("Upload PDFs to preview chunking.")

	with st.expander("Test Retrieval", expanded=False):
		test_query = st.text_input("Debug query", value="", placeholder="Enter a query to test retrieval")
		test_topk = st.slider("Top-k (no threshold filter)", min_value=1, max_value=20, value=5)
		if st.button("Run retrieval test"):
			from rag.retriever import retrieve_relevant_chunks, RetrievalConfig
			cfg = RetrievalConfig(top_k=int(test_topk), similarity_threshold=0.0)
			res = retrieve_relevant_chunks(test_query, embedder, store, cfg)
			if not res.chunks:
				st.warning("No chunks retrieved.")
			else:
				for item in res.chunks:
					st.markdown(f"Score: {item.score:.3f} — {item.chunk.source} (page {item.chunk.page})")
					st.code(item.chunk.text[:600] + ("..." if len(item.chunk.text) > 600 else ""))


def main() -> None:
	st.set_page_config(page_title="RAG with Internet Fallback", layout="wide")
	init_session_state()
	top_k, sim_threshold = sidebar_controls()

	st.title("RAG App: PDFs + Internet Fallback")
	st.caption("Upload PDFs, ask questions, and optionally fallback to web search. Cites sources and shows groundedness.")

	embedder = get_embedder()
	store = get_vector_store(st.session_state.collection_name)
	llm = get_llm(st.session_state.model_backend, st.session_state.gpt4all_model, st.session_state.llama_cpp_model_path)

	with st.expander("Upload and Ingest PDFs", expanded=True):
		num_docs, total_chunks = upload_and_ingest(embedder, store)
		if total_chunks > 0:
			st.success(f"Ingested {num_docs} doc(s), {total_chunks} chunk(s).")

	ask_question_ui(embedder, store, llm, top_k, sim_threshold)
	monitoring_panel(store)
	diagnostics_panel(embedder, store)


if __name__ == "__main__":
	main()


