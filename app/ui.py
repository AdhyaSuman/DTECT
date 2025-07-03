import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
import sys
import os
import base64
import streamlit.components.v1 as components
import html

# Absolute path to the repo root (assuming `ui.py` is in /app)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)
ASSETS_DIR = os.path.join(REPO_ROOT, 'assets')
DATA_DIR = os.path.join(REPO_ROOT, 'data')

# Import functions from the backend
from backend.inference.process_beta import (
    load_beta_matrix,
    get_top_words_over_time,
    load_time_labels
    )
from backend.inference.word_selector import get_interesting_words, get_word_trend
from backend.inference.indexing_utils import load_index
from backend.inference.doc_retriever import (
    load_length_stats,
    get_yearly_counts_for_word,
    deduplicate_docs,
    get_all_documents_for_word_year,
    highlight_words,
    extract_snippet
)
from backend.llm_utils.summarizer import summarize_multiword_docs, ask_multiturn_followup
from backend.llm_utils.label_generator import get_topic_labels
from backend.llm.llm_router import get_llm, list_supported_models
from backend.llm_utils.token_utils import estimate_k_max_from_word_stats

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Page Configuration ---
st.set_page_config(
    page_title="DTECT",
    page_icon="🔍",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/AdhyaSuman/DTECT',
        'Report a bug': "https://github.com/AdhyaSuman/DTECT/issues/new"
    }
)

# Sidebar branding and repo link
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <a href="https://github.com/AdhyaSuman/DTECT" target="_blank">
            <img src="data:image/png;base64,{}" width="180" style="margin-bottom: 18px;">
        </a>
        <hr style="margin-bottom: 0;">
    </div>
    """.format(get_base64_image(os.path.join(ASSETS_DIR, 'Logo_light.png'))),
    unsafe_allow_html=True
)

# 1. Sidebar: Model and Dataset Selection
st.sidebar.title("Configuration")

AVAILABLE_MODELS = ["DTM", "DETM", "CFDTM"]
ENV_VAR_MAP = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Gemini": "GEMINI_API_KEY"
}

def list_datasets(data_dir):
    return sorted([
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ])

with st.sidebar.expander("Select Dataset & Topic Model", expanded=True):
    datasets = list_datasets(DATA_DIR)
    selected_dataset = st.selectbox("Dataset", datasets, help="Choose an available dataset.")
    selected_model = st.selectbox("Model", AVAILABLE_MODELS, help="Select topic model architecture.")

# Check if the dataset has changed and reset session state if it has.
if 'current_dataset' not in st.session_state or st.session_state.current_dataset != selected_dataset:
    st.session_state.current_dataset = selected_dataset
    # List all session state keys that depend on the dataset
    keys_to_clear = [
        "selected_words",
        "interesting_words",
        "word_counts_multiselect",
        "collected_deduplicated_docs",
        "summary",
        "context_for_followup",
        "followup_history"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Rerun the script to apply the clean state
    st.rerun()

# Resolve paths
dataset_path = os.path.join(DATA_DIR, selected_dataset)
model_path = os.path.join(dataset_path, selected_model)
vocab_path = os.path.join(dataset_path, "processed/vocab.txt")
time2id_path = os.path.join(dataset_path, "processed/time2id.txt")
length_stats_path = os.path.join(dataset_path, "processed/length_stats.json")
lemma_map_path = os.path.join(dataset_path, "processed/lemma_to_forms.json")
docs_path = os.path.join(dataset_path, "docs.jsonl")
index_path = os.path.join(dataset_path, "inverted_index.json")
beta_path = os.path.join(model_path, "beta.npy")
label_cache_path = os.path.join(model_path, "topic_label_cache.json")


with st.sidebar.expander("LLM Settings", expanded=True):
    provider = st.selectbox("LLM Provider", options=list(ENV_VAR_MAP.keys()), help="Choose the LLM backend.")
    available_models = list_supported_models(provider)
    model = st.selectbox("LLM Model", options=available_models)
    env_var = ENV_VAR_MAP[provider]
    api_key = os.getenv(env_var)

    if "llm_configured" not in st.session_state:
        st.session_state.llm_configured = False

    if api_key:
        st.session_state.llm_configured = True
    else:
        st.session_state.llm_configured = False
        with st.form(key="api_key_form"):
            entered_key = st.text_input(f"Enter your {provider} API Key", type="password")
            submitted = st.form_submit_button("Submit and Confirm")
            if submitted:
                if entered_key:
                    os.environ[env_var] = entered_key
                    api_key = entered_key
                    st.session_state.llm_configured = True
                    st.rerun()
                else:
                    st.warning("Please enter a key.")

    if not st.session_state.llm_configured:
        st.warning("Please configure your LLM settings in the sidebar.")
        st.stop()

    if api_key and not st.session_state.llm_configured:
        st.session_state.llm_configured = True

    if not api_key:
        st.session_state.llm_configured = False

    if not st.session_state.llm_configured:
        st.warning("Please configure your LLM settings in the sidebar.")
        st.stop()

# Initialize LLM with the provided key
llm = get_llm(provider=provider, model=model, api_key=api_key)

# 3. Load Data
@st.cache_resource
def load_resources(beta_path, vocab_path, docs_path, index_path, time2id_path, length_stats_path, lemma_map_path):
    beta, vocab = load_beta_matrix(beta_path, vocab_path)
    index, docs, lemma_to_forms = load_index(docs_file_path=docs_path, vocab=vocab, index_path=index_path, lemma_map_path=lemma_map_path)
    time_labels = load_time_labels(time2id_path)
    length_stats = load_length_stats(length_stats_path)
    return beta, vocab, index, docs, lemma_to_forms, time_labels, length_stats

# --- Main Title and Paper-aligned Intro ---
st.markdown("""# 🔍 DTECT: Dynamic Topic Explorer & Context Tracker""")

# --- Load resources ---
try:
    beta, vocab, index, docs, lemma_to_forms, time_labels, length_stats = load_resources(
        beta_path,
        vocab_path,
        docs_path,
        index_path,
        time2id_path,
        length_stats_path,
        lemma_map_path
    )
except FileNotFoundError as e:
    st.error(f"Missing required file: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

timestamps = list(range(len(time_labels)))
num_topics = beta.shape[1]
# Estimate max_k based on document length stats and selected LLM
suggested_max_k = estimate_k_max_from_word_stats(length_stats.get("avg_len"), model_name=model, provider=provider)


# ==============================================================================
# 1. 🏷 TOPIC LABELING
# ==============================================================================
st.markdown("## 1️⃣ 🏷️ Topic Labeling")
st.info("Topics are automatically labeled using LLMs by analyzing their temporal word distributions.")

with st.spinner("✨ Generating topic labels... LLM will be used only if labels are not cached."):
    topic_labels = get_topic_labels(beta, vocab, time_labels, llm, label_cache_path)
topic_options = list(topic_labels.values())
selected_topic_label = st.selectbox("Select a Topic", topic_options, help="LLM-generated topic label")
label_to_topic = {v: k for k, v in topic_labels.items()}
selected_topic = label_to_topic[selected_topic_label]

# ==============================================================================
# 2. 💡 INFORMATIVE WORD DETECTION & 📊 TREND VISUALIZATION
# ==============================================================================
st.markdown("---")
st.markdown("## 2️⃣ 💡 Informative Word Detection & 📊 Trend Visualization")
st.info("Explore top/interesting words for each topic, and visualize their trends over time.")

top_n_words = st.slider("Number of Top Words per Topic", min_value=5, max_value=500, value=500)
top_words = get_top_words_over_time(
    beta=beta,
    vocab=vocab,
    topic_id=selected_topic,
    top_n=top_n_words
)

st.write(f"### Top {top_n_words} Words for Topic '{selected_topic_label}' (Ranked):")
scrollable_top_words = "<div style='max-height: 200px; overflow-y: auto; padding: 0 10px;'>"
words_per_col = (top_n_words + 3) // 4
columns = [top_words[i:i+words_per_col] for i in range(0, len(top_words), words_per_col)]
scrollable_top_words += "<div style='display: flex; gap: 20px;'>"
word_rank = 1
for col in columns:
    scrollable_top_words += "<div style='flex: 1;'>"
    for word in col:
        scrollable_top_words += f"<div style='margin-bottom: 4px;'>{word_rank}. {word}</div>"
        word_rank += 1
    scrollable_top_words += "</div>"
scrollable_top_words += "</div></div>"
st.markdown(scrollable_top_words, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

if st.button("💡 Suggest Informative Words", key="suggest_topic_words"):
    top_words = get_top_words_over_time(
        beta=beta,
        vocab=vocab,
        topic_id=selected_topic,
        top_n=top_n_words
    )
    interesting_words = get_interesting_words(beta, vocab, topic_id=selected_topic, restrict_to=top_words)
    st.session_state.interesting_words = interesting_words
    st.session_state.selected_words = interesting_words[:15]  # pre-fill multiselect
    styled_words = " ".join([
        f"<span style='background-color:#e0f7fa; color:#004d40; font-weight:500; padding:4px 8px; margin:4px; border-radius:8px; display:inline-block;'>{w}</span>"
        for w in interesting_words
    ])
    st.markdown(
        f"**Top Informative Words from Topic '{selected_topic_label}':**<br>{styled_words}",
        unsafe_allow_html=True
    )

st.markdown("#### 📈 Plot Word Trends Over Time")
all_word_options = vocab
interesting_words = st.session_state.get("interesting_words", [])

if "selected_words" not in st.session_state:
    st.session_state.selected_words = interesting_words[:15]  # initial default

selected_words = st.multiselect(
    "Select words to visualize trends",
    options=all_word_options,
    default=st.session_state.selected_words,
    key="selected_words"
)
if selected_words:
    fig = go.Figure()
    color_cycle = pc.qualitative.Plotly
    for i, word in enumerate(selected_words):
        trend = get_word_trend(beta, vocab, word, topic_id=selected_topic)
        color = color_cycle[i % len(color_cycle)]
        # --- START: Modify this line ---
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=trend,
            name=word,
            mode='lines+markers',  # Explicitly add markers to the lines
            line=dict(color=color),
            legendgroup=word,
            showlegend=True
        ))
    fig.update_layout(
        title="", 
        xaxis_title="Year", 
        yaxis_title="Importance",
        legend=dict(
            font=dict(
                size=16
            )
        )
    )
    _, chart_col, _ = st.columns([0.2, 0.6, 0.2])
    with chart_col:
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ==============================================================================
# 3. 🔍 DOCUMENT RETRIEVAL & 📃 SUMMARIZATION
# ==============================================================================
st.markdown("---")
st.markdown("## 3️⃣ 🔍 Document Retrieval & 📃 Summarization")
st.info("Retrieve and summarize documents matching selected words and years.")

if selected_words:
    st.markdown("#### 📊 Document Frequency Over Time")
    selected_words_for_counts = st.multiselect(
        "Select word(s) to show document frequencies over time",
        options=selected_words,
        default=selected_words[:3],
        key="word_counts_multiselect"
    )

    if selected_words_for_counts:
        color_cycle = pc.qualitative.Set2
        bar_fig = go.Figure()
        for i, word in enumerate(selected_words_for_counts):
            doc_years, doc_counts = get_yearly_counts_for_word(index=index, word=word)
            bar_fig.add_trace(go.Bar(
                x=doc_years,
                y=doc_counts,
                name=word,
                marker_color=color_cycle[i % len(color_cycle)],
                opacity=0.85
            ))
        bar_fig.update_layout(
            barmode="group",
            title="Document Frequency Over Time",
            xaxis_title="Year",
            yaxis_title="Document Count",
            xaxis=dict(
                tickmode='linear',
                dtick=1,
                tickformat='d'
            ),
            bargap=0.2
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("#### 📄 Inspect Documents for Word-Year Pairs")
    # selected_year = st.slider("Select year", min_value=int(time_labels[0]), max_value=int(time_labels[-1]), key="inspect_year_slider")
    selected_year = st.selectbox(
        "Select year",
        options=time_labels, # Use the list of available time labels (years)
        index=0, # Default to the first year in the list
        key="inspect_year_selectbox"
    )
    collected_docs_raw = []
    for word in selected_words_for_counts:
        docs_for_word_year = get_all_documents_for_word_year(
            index=index,
            docs_file_path=docs_path,
            word=word,
            year=selected_year
        )
        for doc in docs_for_word_year:
            doc["__word__"] = word
        collected_docs_raw.extend(docs_for_word_year)

    if collected_docs_raw:
        st.session_state.collected_deduplicated_docs = deduplicate_docs(collected_docs_raw)
        st.write(f"Found {len(collected_docs_raw)} matching documents, {len(st.session_state.collected_deduplicated_docs)} after deduplication.")

        html_blocks = ""
        for doc in st.session_state.collected_deduplicated_docs:
            word = doc["__word__"]
            full_text = html.escape(doc["text"])
            snippet_text = extract_snippet(doc["text"], word)
            highlighted_snippet = highlight_words(
                snippet_text,
                query_words=selected_words_for_counts,
                lemma_to_forms=lemma_to_forms
            )
            html_blocks += f"""
            <div style="margin-bottom: 14px; padding: 10px; background-color: #fffbe6; border: 1px solid #f0e6cc; border-radius: 6px;">
            <div style="color: #333;"><strong>Match:</strong> {word} | <strong>Doc ID:</strong> {doc['id']} | <strong>Timestamp:</strong> {doc['timestamp']}</div>
            <div style="margin-top: 4px; color: #444;"><em>Snippet:</em> {highlighted_snippet}</div>
            <details style="margin-top: 4px;">
                <summary style="cursor: pointer; color: #007acc;">Show full document</summary>
                <pre style="white-space: pre-wrap; color: #111; background-color: #fffef5; padding: 8px; border: 1px solid #f0e6cc; border-radius: 4px;">{full_text}</pre>
            </details>
            </div>
            """
        min_height = 120
        max_height = 700
        per_doc_height = 130
        dynamic_height = min_height + per_doc_height * max(len(st.session_state.collected_deduplicated_docs) - 1, 0)
        container_height = min(dynamic_height, max_height)
        scrollable_html = f"""
            <div style="overflow-y: auto; padding: 10px; 
                        border: 1px solid #f0e6cc; border-radius: 6px; 
                        background-color: #fffbe6; color: #222;
                        margin-bottom: 0;">
                {html_blocks}
            </div>
        """
        components.html(scrollable_html, height=container_height, scrolling=True)
    else:
        st.warning("No documents found for the selected words and year.")

# ==============================================================================
# 4. 💬 CHAT ASSISTANT (Summary & Follow-up)
# ==============================================================================
st.markdown("---")
st.markdown("## 4️⃣ 💬 Chat Assistant")
st.info("Generate summaries from the inspected documents and ask follow-up questions.")

if "summary" not in st.session_state:
    st.session_state.summary = None
if "context_for_followup" not in st.session_state:
    st.session_state.context_for_followup = ""
if "followup_history" not in st.session_state:
    st.session_state.followup_history = []

# MMR K selection
st.markdown(f"**Max documents for summarization (k):**")
st.markdown(f"The suggested maximum number of documents for summarization (k) based on the average document length and the selected LLM is **{suggested_max_k}**.")
mmr_k = st.slider(
    "Select the maximum number of documents (k) for MMR (Maximum Marginal Relevance) selection for summarization.",
    min_value=1,
    max_value=20, # Set a reasonable max for k, can be adjusted
    value=min(suggested_max_k, 20), # Use suggested_max_k as default, capped at 20
    help="This value determines how many relevant and diverse documents will be selected for summarization."
)

if st.button("📃 Summarize These Documents"):
    if st.session_state.get("collected_deduplicated_docs"):
        st.session_state.summary = None
        st.session_state.context_for_followup = ""
        st.session_state.followup_history = []
        with st.spinner("Selecting and summarizing documents..."):
            summary, mmr_docs = summarize_multiword_docs(
                selected_words_for_counts,
                selected_year,
                st.session_state.collected_deduplicated_docs,
                llm,
                k=mmr_k
            )
            st.session_state.summary = summary
            st.session_state.context_for_followup = "\n".join(
                f"Document {i+1}:\n{doc.page_content.strip()}" for i, doc in enumerate(mmr_docs)
            )
            st.session_state.followup_history.append(
                {"role": "user", "content": f"Please summarize the context of the words '{', '.join(selected_words_for_counts)}' in {selected_year} based on the provided documents."}
            )
            st.session_state.followup_history.append(
                {"role": "assistant", "content": st.session_state.summary}
            )
        st.success(f"✅ Summary generated from {len(mmr_docs)} MMR-selected documents.")
    else:
        st.warning("⚠️ No documents collected to summarize. Please inspect some documents first.")

if st.session_state.summary:
    st.markdown(f"**Summary for words `{', '.join(selected_words_for_counts)}` in `{selected_year}`:**")
    st.write(st.session_state.summary)

    if st.checkbox("💬 Ask follow-up questions about this summary", key="enable_followup"):
        with st.expander("View the documents used for this conversation"):
            st.text_area("Context Documents", st.session_state.context_for_followup, height=200)
        st.info("Ask a question based on the summary and the documents above.")
        for msg in st.session_state.followup_history[2:]:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])
        if user_query := st.chat_input("Ask a follow-up question..."):
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_query)
            st.session_state.followup_history.append({"role": "user", "content": user_query})
            with st.spinner("Thinking..."):
                followup_response = ask_multiturn_followup(
                    history=st.session_state.followup_history,
                    question=user_query,
                    llm=llm,
                    context_texts=st.session_state.context_for_followup
                )
            st.session_state.followup_history.append({"role": "assistant", "content": followup_response})
            if followup_response.startswith("[Error"):
                st.error(followup_response)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(followup_response)
            st.rerun()