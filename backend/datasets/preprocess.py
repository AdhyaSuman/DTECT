import json
import os
import numpy as np
from collections import OrderedDict
import tempfile
import gensim.downloader
from tqdm import tqdm
from backend.datasets.utils.logger import Logger
import scipy.sparse
from gensim.models.phrases import Phrases, Phraser
from typing import List, Union
from octis.preprocessing.preprocessing import Preprocessing

logger = Logger("WARNING")

class Preprocessor:
    def __init__(self,
                 docs_jsonl_path: str,
                 output_folder: str,
                 use_partition: bool = False,
                 use_bigrams: bool = False,
                 min_count_bigram: int = 5,
                 threshold_bigram: int = 10,
                 remove_punctuation: bool = True,
                 lemmatize: bool = True,
                 stopword_list: Union[str, List[str]] = None,
                 min_chars: int = 3,
                 min_words_docs: int = 10,
                 min_df: Union[int, float] = 0.0,
                 max_df: Union[int, float] = 1.0,
                 max_features: int = None,
                 language: str = 'english'):
        
        self.docs_jsonl_path = docs_jsonl_path
        self.output_folder = output_folder
        self.use_partition = use_partition
        self.use_bigrams = use_bigrams
        self.min_count_bigram = min_count_bigram
        self.threshold_bigram = threshold_bigram

        os.makedirs(self.output_folder, exist_ok=True)

        self.preprocessing_params = {
            'remove_punctuation': remove_punctuation,
            'lemmatize': lemmatize,
            'stopword_list': stopword_list,
            'min_chars': min_chars,
            'min_words_docs': min_words_docs,
            'min_df': min_df,
            'max_df': max_df,
            'max_features': max_features,
            'language': language
        }
        self.preprocessor_octis = Preprocessing(**self.preprocessing_params)

    def _load_data_to_temp_files(self):
        """Loads data from JSONL and writes to temporary files for OCTIS preprocessor."""
        raw_texts = []
        raw_timestamps = []
        raw_labels = []
        has_labels = False

        with open(self.docs_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Remove newlines from text
                clean_text = data.get('text', '').replace('\n', ' ').replace('\r', ' ')
                clean_text = " ".join(clean_text.split())
                raw_texts.append(clean_text)
                raw_timestamps.append(data.get('timestamp', ''))
                label = data.get('label', '')
                if label:
                    has_labels = True
                raw_labels.append(label)
        
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        temp_docs_path = os.path.join(temp_dir, "temp_docs.txt")
        temp_labels_path = None

        with open(temp_docs_path, 'w', encoding='utf-8') as f_docs:
            for text in raw_texts:
                f_docs.write(f"{text}\n")
        
        if has_labels:
            temp_labels_path = os.path.join(temp_dir, "temp_labels.txt")
            with open(temp_labels_path, 'w', encoding='utf-8') as f_labels:
                for label in raw_labels:
                    f_labels.write(f"{label}\n")

        print(f"Loaded {len(raw_texts)} raw documents and created temporary files in {temp_dir}.")
        return raw_texts, raw_timestamps, raw_labels, temp_docs_path, temp_labels_path, temp_dir

    def _make_word_embeddings(self, vocab):
        """
        Generates word embeddings for the given vocabulary using GloVe.
        For n-grams (e.g., "wordA_wordB", "wordX_wordY_wordZ" for n>=2),
        the resultant embedding is the sum of the embeddings of its constituent
        single words (wordA + wordB + ...).
        """
        print("Loading GloVe word embeddings...")
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
        
        # Initialize word_embeddings matrix with zeros.
        # This ensures that words not found (single or n-gram constituents)
        # will have a zero vector embedding.
        word_embeddings = np.zeros((len(vocab), glove_vectors.vectors.shape[1]), dtype=np.float32)

        num_found = 0

        try:
            # Using a set for key_word_list for O(1) average time complexity lookup
            key_word_list = set(glove_vectors.index_to_key)
        except AttributeError: # For older gensim versions
            key_word_list = set(glove_vectors.index2word)

        print("Generating word embeddings for vocabulary (including n-grams)...")
        for i, word in enumerate(tqdm(vocab, desc="Processing vocabulary words")):
            if '_' in word: # Check if it's a potential n-gram (n >= 2)
                parts = word.split('_')
                
                # Check if *all* constituent words are present in GloVe
                all_parts_in_glove = True
                for part in parts:
                    if part not in key_word_list:
                        all_parts_in_glove = False
                        break # One part not found, stop checking
                
                if all_parts_in_glove:
                    # If all parts are found, sum their embeddings
                    resultant_vector = np.zeros(glove_vectors.vectors.shape[1], dtype=np.float32)
                    for part in parts:
                        resultant_vector += glove_vectors[part]
                    
                    word_embeddings[i] = resultant_vector
                    num_found += 1
                # Else: one or more constituent words not found, embedding remains zero
            else: # It's a single word (n=1)
                if word in key_word_list:
                    word_embeddings[i] = glove_vectors[word]
                    num_found += 1
                # Else: single word not found, embedding remains zero

        logger.info(f'Number of found embeddings (including n-grams): {num_found}/{len(vocab)}')
        return word_embeddings # Return as dense NumPy array
    
    
    def _save_doc_length_stats(self, filepath: str, output_path: str):
        doc_lengths = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = line.strip()
                    if doc:
                        doc_lengths.append(len(doc))
        except Exception as e:
            print(f"Error processing '{filepath}': {e}")
            return

        if not doc_lengths:
            print(f"No documents found in '{filepath}'.")
            return

        stats = {
            "avg_len": float(np.mean(doc_lengths)),
            "std_len": float(np.std(doc_lengths)),
            "max_len": int(np.max(doc_lengths)),
            "min_len": int(np.min(doc_lengths)),
            "num_docs": int(len(doc_lengths))
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        print(f"Saved document length stats to: {output_path}")
        

    def preprocess(self):
        print("Loading data and creating temporary files for OCTIS...")
        _, raw_timestamps, _, temp_docs_path, temp_labels_path, temp_dir = \
            self._load_data_to_temp_files()

        print("Starting OCTIS pre-processing using file paths and specified parameters...")
        octis_dataset = self.preprocessor_octis.preprocess_dataset(
            documents_path=temp_docs_path,
            labels_path=temp_labels_path
        )
        
        # Clean up temporary files immediately
        os.remove(temp_docs_path)
        if temp_labels_path:
            os.remove(temp_labels_path)
        os.rmdir(temp_dir)
        print(f"Temporary files in {temp_dir} cleaned up.")
        
        # --- Proxy: Save __original_indexes and then manually load it ---
        temp_indexes_dir = tempfile.mkdtemp()
        temp_indexes_file = os.path.join(temp_indexes_dir, "temp_original_indexes.txt")
        
        print(f"Saving __original_indexes to {temp_indexes_file}...")
        octis_dataset._save_document_indexes(temp_indexes_file)
        
        # Manually load the indexes from the file
        original_indexes_after_octis = []
        with open(temp_indexes_file, 'r') as f_indexes:
            for line in f_indexes:
                original_indexes_after_octis.append(int(line.strip())) # Read as int
        
        # Clean up the temporary indexes file and its directory
        os.remove(temp_indexes_file)
        os.rmdir(temp_indexes_dir) 
        print("Temporary indexes file cleaned up.")
        # --- End Proxy ---

        # Get processed data from OCTIS Dataset object
        processed_corpus_octis_list = octis_dataset.get_corpus() # List of list of tokens
        processed_labels_octis = octis_dataset.get_labels() # List of labels

        print("Max index in original_indexes_after_octis:", max(original_indexes_after_octis))
        print("Length of raw_timestamps:", len(raw_timestamps))
        
        # Filter timestamps based on documents that survived OCTIS preprocessing
        filtered_timestamps = [raw_timestamps[i] for i in original_indexes_after_octis]

        print(f"OCTIS preprocessing complete. {len(processed_corpus_octis_list)} documents remaining.")
        
        if self.use_bigrams:
            print("Generating bigrams with Gensim...")
            phrases = Phrases(processed_corpus_octis_list, min_count=self.min_count_bigram, threshold=self.threshold_bigram)
            bigram_phraser = Phraser(phrases)
            bigrammed_corpus_list = [bigram_phraser[doc] for doc in processed_corpus_octis_list]
            print("Bigram generation complete.")
        else:
            print("Skipping bigram generation as 'use_bigrams' is False.")
            bigrammed_corpus_list = processed_corpus_octis_list # Use the original processed list


        # Convert back to list of strings for easier handling if needed later, but keep as list of lists for BOW
        bigrammed_texts_for_file = [" ".join(doc) for doc in bigrammed_corpus_list]
        print("Bigram generation complete.")

        # Build Vocabulary from OCTIS output (after bigrams)
        # We need a flat list of all tokens to build the vocabulary
        all_tokens = [token for doc in bigrammed_corpus_list for token in doc]
        vocab = sorted(list(set(all_tokens))) # Sorted unique words form the vocabulary
        word_to_id = {word: i for i, word in enumerate(vocab)}

        # Create BOW matrix manually
        print("Creating Bag-of-Words representations...")
        rows, cols, data = [], [], []
        for i, doc_tokens in enumerate(bigrammed_corpus_list):
            doc_word_counts = {}
            for token in doc_tokens:
                if token in word_to_id: # Ensure token is in our final vocab
                    doc_word_counts[word_to_id[token]] = doc_word_counts.get(word_to_id[token], 0) + 1
            for col_id, count in doc_word_counts.items():
                rows.append(i)
                cols.append(col_id)
                data.append(count)
        
        # Shape is (num_documents, vocab_size)
        bow_matrix = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(len(bigrammed_corpus_list), len(vocab)))
        print("Bag-of-Words complete.")
        
        # Handle partitioning if required
        if self.use_partition:
            num_docs = len(bigrammed_corpus_list)
            train_size = int(0.8 * num_docs)
            
            train_texts = bigrammed_texts_for_file[:train_size]
            train_bow_matrix = bow_matrix[:train_size]
            train_timestamps = filtered_timestamps[:train_size]
            train_labels = processed_labels_octis[:train_size] if processed_labels_octis else []

            test_texts = bigrammed_texts_for_file[train_size:]
            test_bow_matrix = bow_matrix[train_size:]
            test_timestamps = filtered_timestamps[train_size:]
            test_labels = processed_labels_octis[train_size:] if processed_labels_octis else []

        else:
            train_texts = bigrammed_texts_for_file
            train_bow_matrix = bow_matrix
            train_timestamps = filtered_timestamps
            train_labels = processed_labels_octis
            test_texts = []
            test_timestamps = []
            test_labels = []

        # Generate word embeddings using the provided function
        word_embeddings = self._make_word_embeddings(vocab)

        # Process timestamps to 0, 1, 2...T and create time2id.txt
        print("Processing timestamps...")
        unique_timestamps = sorted(list(set(train_timestamps + test_timestamps)))
        time_to_id = {timestamp: i for i, timestamp in enumerate(unique_timestamps)}

        train_times_ids = [time_to_id[ts] for ts in train_timestamps]
        test_times_ids = [time_to_id[ts] for ts in test_timestamps] if self.use_partition else []
        print("Timestamps processed.")

        # Save files
        print(f"Saving preprocessed files to {self.output_folder}...")
        
        # 1. vocab.txt
        with open(os.path.join(self.output_folder, "vocab.txt"), "w", encoding="utf-8") as f:
            for word in vocab:
                f.write(f"{word}\n")

        # 2. train_texts.txt
        train_text_path = os.path.join(self.output_folder, "train_texts.txt")
        with open(train_text_path, "w", encoding="utf-8") as f:
            for doc in train_texts:
                f.write(f"{doc}\n")
        
        # Save document length stats
        doc_stats_path = os.path.join(self.output_folder, "length_stats.json")
        self._save_doc_length_stats(train_text_path, doc_stats_path)

        # 3. train_bow.npz
        scipy.sparse.save_npz(os.path.join(self.output_folder, "train_bow.npz"), train_bow_matrix)

        # 4. word_embeddings.npz
        sparse_word_embeddings = scipy.sparse.csr_matrix(word_embeddings)
        scipy.sparse.save_npz(os.path.join(self.output_folder, "word_embeddings.npz"), sparse_word_embeddings)
        
        # 5. train_labels.txt (if labels exist)
        if train_labels: 
            with open(os.path.join(self.output_folder, "train_labels.txt"), "w", encoding="utf-8") as f:
                for label in train_labels:
                    f.write(f"{label}\n")

        # 6. train_times.txt
        with open(os.path.join(self.output_folder, "train_times.txt"), "w", encoding="utf-8") as f:
            for time_id in train_times_ids:
                f.write(f"{time_id}\n")

        # Files for test set (if use_partition=True)
        if self.use_partition:
            # 7. test_bow.npz
            scipy.sparse.save_npz(os.path.join(self.output_folder, "test_bow.npz"), test_bow_matrix)

            # 8. test_texts.txt
            with open(os.path.join(self.output_folder, "test_texts.txt"), "w", encoding="utf-8") as f:
                for doc in test_texts:
                    f.write(f"{doc}\n")

            # 9. test_labels.txt (if labels exist)
            if test_labels: 
                with open(os.path.join(self.output_folder, "test_labels.txt"), "w", encoding="utf-8") as f:
                    for label in test_labels:
                        f.write(f"{label}\n")

            # 10. test_times.txt
            with open(os.path.join(self.output_folder, "test_times.txt"), "w", encoding="utf-8") as f:
                for time_id in test_times_ids:
                    f.write(f"{time_id}\n")
        
        # 11. time2id.txt
        sorted_time_to_id = OrderedDict(sorted(time_to_id.items(), key=lambda item: item[1]))
        with open(os.path.join(self.output_folder, "time2id.txt"), "w", encoding="utf-8") as f:
            json.dump(sorted_time_to_id, f, indent=4) 
        
        print("All files saved successfully.")