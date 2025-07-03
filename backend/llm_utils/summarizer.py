import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
    
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- MMR Utilities ---
def build_mmr_index(docs):
    texts = [doc['text'] for doc in docs if 'text' in doc]
    documents = [Document(page_content=text) for text in texts]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([doc.page_content for doc in documents], convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index, embeddings, documents

def get_mmr_sample(model, index, embeddings, documents, query, k=15, lambda_mult=0.7):
    if len(documents) == 0:
        print("Warning: No documents available, returning empty list.")
        return []

    if len(documents) <= k:
        print(f"Warning: Only {len(documents)} documents available, returning all.")
        return documents

    else:
        query_vec = model.encode(query, convert_to_numpy=True)
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Get candidate indices from FAISS (k * 4 or less if not enough documents)
        num_candidates = min(k * 4, len(documents))
        D, I = index.search(np.expand_dims(query_vec, axis=0), num_candidates)
        candidate_idxs = list(I[0])

        selected = []
        while len(selected) < k and candidate_idxs:
            if not selected:
                selected.append(candidate_idxs.pop(0))
                continue

            mmr_scores = []
            for idx in candidate_idxs:
                relevance = cosine_similarity([query_vec], [embeddings[idx]])[0][0]
                diversity = max([
                    cosine_similarity([embeddings[idx]], [embeddings[sel]])[0][0]
                    for sel in selected
                ])
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity
                mmr_scores.append((idx, mmr_score))

            next_best = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(next_best)
            candidate_idxs.remove(next_best)

        return [documents[i] for i in selected]


# --- Summarization ---
def summarize_docs(word, timestamp, docs, llm, k):
    if not docs:
        return "No documents available for this word at this time.", [], 0

    try:
        model, index, embeddings, documents = build_mmr_index(docs)
        mmr_docs = get_mmr_sample(model, index, embeddings, documents, query=word, k=k)

        context_texts = "\n".join(f"- {doc.page_content}" for doc in mmr_docs)

        prompt_template = ChatPromptTemplate.from_template(
            "Given the following documents from {timestamp} containing the word '{word}', "
            "identify the key themes or distinct discussion points that were prevalent during that time. "
            "Do NOT describe each bullet in detail. Be concise. Each bullet should be a short phrase or sentence "
            "capturing a unique, non-overlapping theme. Avoid any elaboration, examples, or justification.\n\n"
            "Return no more than 5–7 bullets.\n\n"
            "{context_texts}\n\nSummary:"
        )

        chain = prompt_template | llm
        summary = chain.invoke({
            "word": word,
            "timestamp": timestamp,
            "context_texts": context_texts
        }).content.strip()

        return summary, mmr_docs

    except Exception as e:
        return f"[Error summarizing: {e}]", [], 0


def summarize_multiword_docs(words, timestamp, docs, llm, k):
    if not docs:
        return "No common documents available for these words at this time.", []

    try:
        model, index, embeddings, documents = build_mmr_index(docs)
        query = " ".join(words)
        mmr_docs = get_mmr_sample(model, index, embeddings, documents, query=query, k=k)

        context_texts = "\n".join(f"- {doc.page_content}" for doc in mmr_docs)

        prompt_template = ChatPromptTemplate.from_template(
            "Given the following documents from {timestamp} that all mention the words: '{word_list}', "
            "identify the key themes or distinct discussion points that were prevalent during that time. "
            "Do NOT describe each bullet in detail. Be concise. Each bullet should be a short phrase or sentence "
            "capturing a unique, non-overlapping theme. Avoid any elaboration, examples, or justification.\n\n"
            "Return no more than 5–7 bullets.\n\n"
            "{context_texts}\n\n"
            "Concise Thematic Summary:"
        )

        chain = prompt_template | llm
        summary = chain.invoke({
            "word_list": ", ".join(words),
            "timestamp": timestamp,
            "context_texts": context_texts
        }).content.strip()

        return summary, mmr_docs

    except Exception as e:
        return f"[Error summarizing: {e}]", []


# --- Follow-up Question Handler (Improved) ---
def ask_multiturn_followup(history: list, question: str, llm, context_texts: str) -> str:
    """
    Handles multi-turn follow-up questions based on a provided set of documents.

    This function now REQUIRES context_texts to be provided, ensuring the LLM
    is always grounded in the source documents for follow-up questions.

    Args:
        history (list): A list of dictionaries representing the conversation history
                        (e.g., [{"role": "user", "content": "..."}]).
        question (str): The user's new follow-up question.
        llm: The initialized language model instance.
        context_texts (str): A single string containing all the numbered documents
                             for context.

    Returns:
        str: The AI's response to the follow-up question.
    """
    try:
        # 1. Reconstruct conversation memory from the history provided from the UI
        memory = ConversationBufferMemory(return_messages=True)
        for turn in history:
            if turn["role"] == "user":
                memory.chat_memory.add_user_message(turn["content"])
            elif turn["role"] == "assistant":
                memory.chat_memory.add_ai_message(turn["content"])

        # 2. Define the system instruction that grounds the LLM
        system_instruction = (
            "You are an assistant answering questions strictly based on the provided sample documents below. "
            "Your memory contains the previous turns of this conversation. "
            "If the answer is not clearly available in the text, respond with: "
            "'The information is not available in the documents provided.'\n\n"
        )

        # 3. Create the full prompt. No more conditional logic, as context is required.
        #    The `ConversationChain` will automatically use the memory, so we only need
        #    to provide the current input, which includes the grounding documents.
        full_prompt = (
            f"{system_instruction}"
            f"--- DOCUMENTS ---\n{context_texts.strip()}\n\n"
            f"--- QUESTION ---\n{question}"
        )

        # 4. Create and run the conversation chain
        conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
        response = conversation.predict(input=full_prompt)
        
        return response.strip()

    except Exception as e:
        # Good practice to log the full exception for easier debugging
        print(f"[ERROR] in ask_multiturn_followup: {e}")
        return f"[Error during multi-turn follow-up. Please check the logs.]"