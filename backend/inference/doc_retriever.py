import html
import json
import re
import os
from hashlib import md5

def deduplicate_docs(collected_docs):
    seen = set()
    unique_docs = []
    for doc in collected_docs:
        # Prefer unique ID if available
        key = doc.get("id", md5(doc["text"].encode()).hexdigest())
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs

def load_length_stats(length_stats_path):
    """
    Loads length statistics from a JSON file for a given model path.

    Args:
        path (str): Path to the model directory containing 'length_stats.json'.

    Returns:
        dict: A dictionary containing document length statistics.
    """   
    if not os.path.exists(length_stats_path):
        raise FileNotFoundError(f"'length_stats.json' not found at: {length_stats_path}")
    
    with open(length_stats_path, "r") as f:
        length_stats = json.load(f)
    
    return length_stats

def get_yearly_counts_for_word(index, word):
    if word not in index:
        print(f"[ERROR] Word '{word}' not found in index.")
        return [], []

    year_counts = index[word]
    sorted_items = sorted((int(year), len(doc_ids)) for year, doc_ids in year_counts.items())
    years, counts = zip(*sorted_items) if sorted_items else ([], [])
    return list(years), list(counts)


def get_all_documents_for_word_year(index, docs_file_path, word, year):
    """
    Returns all full documents (text + metadata) that contain a given word in a given year.

    Parameters:
        index (dict): Inverted index.
        docs_file_path (str): Path to original jsonl corpus.
        word (str): Word (unigram or bigram).
        year (int): Year to retrieve docs for.

    Returns:
        List[Dict]: List of documents with 'id', 'timestamp', and 'text'.
    """
    year = int(year)

    if word not in index or year not in index[word]:
        return []

    doc_ids = set(index[word][year])
    results = []

    try:
        with open(docs_file_path, 'r', encoding='utf-8') as f:
            for doc_id, line in enumerate(f):
                if doc_id in doc_ids:
                    doc = json.loads(line)
                    results.append({
                        "id": doc_id,
                        "timestamp": doc.get("timestamp", "N/A"),
                        "text": doc["text"]
                    })
    except Exception as e:
        print(f"[ERROR] Could not load documents: {e}")

    return results


def get_documents_with_all_words_for_year(index, docs_path, words, year):
    doc_sets = []
    all_doc_occurrences = {}

    for word in words:
        word_docs = get_all_documents_for_word_year(index, docs_path, word, year)
        doc_sets.append(set(doc["id"] for doc in word_docs))
        for doc in word_docs:
            all_doc_occurrences.setdefault(doc["id"], doc)

    common_doc_ids = set.intersection(*doc_sets) if doc_sets else set()
    return [all_doc_occurrences[doc_id] for doc_id in common_doc_ids]


def get_intersection_doc_counts_by_year(index, docs_path, words, all_years):
    year_counts = {}
    for y in all_years:
        docs = get_documents_with_all_words_for_year(index, docs_path, words, y)
        year_counts[y] = len(docs)
    return year_counts


def extract_snippet(text, query, window=30):
    """
    Return a short snippet around the first occurrence of the query word.
    """
    pattern = re.compile(re.escape(query.replace('_', ' ')), re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return text[:200] + "..."

    start = max(match.start() - window, 0)
    end = min(match.end() + window, len(text))
    snippet = text[start:end].strip()

    return f"...{snippet}..."

def highlight(text, query, highlight_color="#FFD54F"):
    """
    Highlight all instances of the query term in text using a colored <mark> tag.
    """
    escaped_query = re.escape(query.replace('_', ' '))
    pattern = re.compile(f"({escaped_query})", flags=re.IGNORECASE)

    def replacer(match):
        matched_text = html.escape(match.group(1))
        return f"<mark style='background-color:{highlight_color}; color:black;'>{matched_text}</mark>"

    return pattern.sub(replacer, html.escape(text))

def highlight_words(text, query_words, highlight_color="#24F31D", lemma_to_forms=None):
    """
    Highlight all surface forms of each query lemma in the text using a colored <mark> tag.

    Args:
        text (str): The input raw document text.
        query_words (List[str]): Lemmatized query tokens to highlight.
        highlight_color (str): Color to use for highlighting.
        lemma_to_forms (Dict[str, Set[str]]): Maps a lemma to its surface forms.
    """
    # Escape HTML special characters first
    escaped_text = html.escape(text)

    # Expand query words to include all surface forms
    expanded_forms = set()
    for lemma in query_words:
        # Also handle bigrams passed directly
        expanded_forms.add(lemma)
        if lemma_to_forms and lemma in lemma_to_forms:
            expanded_forms.update(lemma_to_forms[lemma])

    # Sort by length to avoid partial overlaps (e.g., "run" before "running")
    sorted_queries = sorted(list(expanded_forms), key=lambda w: -len(w))

    for word in sorted_queries:
        # Prepare the word for regex: replace underscores with spaces for bigrams
        search_term = word.replace('_', ' ')
        # Match full word/phrase, case insensitive
        pattern = re.compile(rf'\b({re.escape(search_term)})\b', flags=re.IGNORECASE)

        def replacer(match):
            matched_text = match.group(1)
            return f"<mark style='background-color:{highlight_color}; color:black;'>{matched_text}</mark>"

        escaped_text = pattern.sub(replacer, escaped_text)

    return escaped_text

def get_docs_by_ids(docs_file_path, doc_ids):
    """
    Efficiently retrieves specific documents from a .jsonl file by their line number (ID).

    This function reads the file line-by-line and only parses the lines that match
    the requested document IDs, avoiding loading the entire file into memory.

    Args:
        docs_file_path (str): The path to the documents.jsonl file.
        doc_ids (list or set): A collection of document IDs (0-indexed line numbers) to retrieve.

    Returns:
        list[dict]: A list of document dictionaries that were found. Each dictionary
                    is augmented with an 'id' key corresponding to its line number.
    """
    # Use a set for efficient O(1) lookups.
    doc_ids_to_find = set(doc_ids)
    found_docs = {}

    if not doc_ids_to_find:
        return []

    try:
        with open(docs_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # If the current line number is one we're looking for
                if i in doc_ids_to_find:
                    try:
                        doc = json.loads(line)
                        # Explicitly add the line number as the 'id'
                        doc['id'] = i
                        found_docs[i] = doc
                        # Optimization: stop reading the file once all docs are found
                        if len(found_docs) == len(doc_ids_to_find):
                            break
                    except json.JSONDecodeError:
                        # Skip malformed lines but inform the user
                        print(f"[WARNING] Skipping malformed JSON on line {i+1} in {docs_file_path}")
                        continue

    except FileNotFoundError:
        print(f"[ERROR] Document file not found at: {docs_file_path}")
        return []
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while reading documents: {e}")
        return []

    # Return the documents in the same order as the original doc_ids list
    # This ensures consistency for downstream processing.
    return [found_docs[doc_id] for doc_id in doc_ids if doc_id in found_docs]