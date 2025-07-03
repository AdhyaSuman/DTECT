
<p align="center">
  <img src="https://github.com/AdhyaSuman/DTECT/blob/main/assets/Logo_light.png" width="400"/>
</p>

-----

## 👋 Introduction

> **DTECT (Dynamic Topic Explorer & Context Tracker)** is an end-to-end, open-source system designed to streamline the entire process of dynamic topic modeling.

-----

## 🚀 Live Demo & Video

  * **Interactive Demo:** Try DTECT live on Hugging Face Spaces\!

      * [https://huggingface.co/spaces/AdhyaSuman/DTECT](https://huggingface.co/spaces/AdhyaSuman/DTECT)

* **Demo Video:** Watch a walkthrough of DTECT's features.



    [![Watch the Demo Video](https://img.youtube.com/vi/B8nNfxFoJAU/0.jpg)](https://www.youtube.com/watch?v=B8nNfxFoJAU)

-----

## 💻 Code Examples

### DTECT Preprocessing Pipeline

Here is an example of how to use the DTECT preprocessing pipeline for a custom dataset:

```python
from backend.datasets.preprocess import Preprocessor
from nltk.corpus import stopwords
import os

dataset_dir = '../data/Sample_data/'
stop_words = stopwords.words('english')

preprocessor = Preprocessor(
    docs_jsonl_path=dataset_dir + 'docs.jsonl',
    output_folder=os.path.join(dataset_dir, 'processed'),
    use_partition=False,
    min_count_bigram=5,
    threshold_bigram=20,
    remove_punctuation=True,
    lemmatize=True,
    stopword_list=stop_words,
    min_chars=3,
    min_words_docs=3,
)
preprocessor.preprocess()
```

### Training and Evaluation Pipeline (with CFDTM)

This snippet shows an example of the training and evaluation pipeline using the CFDTM model:

```python
from backend.datasets import dynamic_dataset
from backend.models.CFDTM.CFDTM import CFDTM
from backend.models.dynamic_trainer import DynamicTrainer
from backend.evaluation.eval import TopicQualityAssessor

# Load dataset
data = dynamic_dataset.DynamicDataset('../data/Sample_data/processed')

# Initialize model
model = CFDTM(
    vocab_size=data.vocab_size,
    num_times=data.num_times,
    num_topics=20,
    pretrained_WE=data.pretrained_WE,
    train_time_wordfreq=data.train_time_wordfreq
).to("cuda")

# Train model
trainer = DynamicTrainer(model, data)
top_words, _ = trainer.train()
top_words_list = [[topic.split() for topic in timestamp] for timestamp in top_words]
train_corpus = [doc.split() for doc in data.train_texts]

# Evaluation
assessor = TopicQualityAssessor(
    topics=top_words_list,
    train_texts=train_corpus,
    topn=10,
    coherence_type='c_npmi'
)
summary = assessor.get_dtq_summary()
```

-----

## 📁 Repository Structure

```
├── app
│   └── ui.py
├── assets
│   ├── Logo_dark.png
│   └── Logo_light.png
├── backend
│   ├── datasets
│   │   ├── data
│   │   │   ├── download.py
│   │   │   └── file_utils.py
│   │   ├── dynamic_dataset.py
│   │   ├── preprocess.py
│   │   └── utils
│   │       ├── logger.py
│   │       └── _utils.py
│   ├── evaluation
│   │   ├── CoherenceModel_ttc.py
│   │   ├── eval.py
│   │   ├── topic_coherence.py
│   │   └── topic_diversity.py
│   ├── inference
│   │   ├── doc_retriever.py
│   │   ├── indexing_utils.py
│   │   ├── peak_detector.py
│   │   ├── process_beta.py
│   │   └── word_selector.py
│   ├── __init__.py
│   ├── llm
│   │   ├── custom_gemini.py
│   │   ├── custom_mistral.py
│   │   └── llm_router.py
│   ├── llm_utils
│   │   ├── label_generator.py
│   │   ├── summarizer.py
│   │   └── token_utils.py
│   └── models
│       ├── CFDTM
│       │   ├── CFDTM.py
│       │   ├── Encoder.py
│       │   ├── ETC.py
│       │   ├── __init__.py
│       │   └── UWE.py
│       ├── DETM.py
│       ├── DTM_trainer.py
│       └── dynamic_trainer.py
├── data
│   └── Sample_data
│       ├── CFDTM
│       │   └── beta.npy
│       ├── docs.jsonl
│       └── processed
│           ├── length_stats.json
│           ├── time2id.txt
│           ├── train_bow.npz
│           ├── train_texts.txt
│           ├── train_times.txt
│           ├── vocab.txt
│           └── word_embeddings.npz
├── environment.yml
├── LICENSE
├── main.py
└── requirements.txt
```

-----

## 📚 Supporting Resources

We list below the datasets, codebases, and evaluation resources referenced or integrated into DTECT:

#### Datasets

- [ACL Anthology](https://aclanthology.org/)
- [UN General Debates](https://www.kaggle.com/datasets/unitednations/un-general-debates)
- [TCPD-IPD Finance](https://tcpd.ashoka.edu.in/question-hour/)

#### Dynamic Topic Modeling Codebases

- [DTM Trainer](https://github.com/bobxwu/TopMost/blob/main/topmost/trainers/dynamic/DTM_trainer.py)
- [DETM Model](https://github.com/bobxwu/TopMost/blob/main/topmost/models/dynamic/DETM.py)
- [CFDTM Model](https://github.com/bobxwu/TopMost/tree/main/topmost/models/dynamic/CFDTM)

#### Preprocessing Toolkit

- [OCTIS](https://github.com/MIND-Lab/OCTIS)

#### Evaluation Metrics

  * **Evaluating Dynamic Topic Models:** [https://github.com/CharuJames/Evaluating-Dynamic-Topic-Models](https://github.com/CharuJames/Evaluating-Dynamic-Topic-Models)

---

## 🙏 Acknowledgements

We would like to acknowledge the following open-source projects that were instrumental in the development of DTECT:

1.  **TopMost Toolkit:** [https://github.com/bobxwu/TopMost](https://github.com/bobxwu/TopMost)
2.  **OCTIS:** [https://github.com/MIND-Lab/OCTIS](https://github.com/MIND-Lab/OCTIS)

---
