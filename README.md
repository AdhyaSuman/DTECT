
<p align="center">
  <img src="https://github.com/AdhyaSuman/DTECT/blob/main/assets/Logo_light.png" width="200"/>
</p>


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
---

## 📁 Repository Structure

```
├── app
│   └── ui.py
├── assets
├── backend
│   ├── datasets
│   ├── evaluation
│   ├── inference
│   ├── llm
│   ├── llm_utils
│   └── models
├── data
│   └── Sample_data
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

- [DTM](https://github.com/bobxwu/TopMost/blob/main/topmost/trainers/dynamic/DTM_trainer.py)
- [DETM](https://github.com/bobxwu/TopMost/blob/main/topmost/models/dynamic/DETM.py)
- [CFDTM](https://github.com/bobxwu/TopMost/tree/main/topmost/models/dynamic/CFDTM)

#### Preprocessing Toolkit

- [OCTIS](https://github.com/MIND-Lab/OCTIS)

#### Evaluation Metrics

  * **Evaluating Dynamic Topic Models:** [https://github.com/CharuJames/Evaluating-Dynamic-Topic-Models](https://github.com/CharuJames/Evaluating-Dynamic-Topic-Models)

---

## 🙏 Acknowledgements

We would like to acknowledge the following open-source projects that were instrumental in the development of DTECT:

> 🔍 **TopMost Toolkit**
> [https://github.com/bobxwu/TopMost](https://github.com/bobxwu/TopMost)
> 📌 **Reference:** Xiaobao Wu, Fengjun Pan, and Anh Tuan Luu. 2024. Towards the TopMost: A Topic Modeling System Toolkit. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 31–41, Bangkok, Thailand. Association for Computational Linguistics.

<br>

> 📦 **OCTIS**
> [https://github.com/MIND-Lab/OCTIS](https://github.com/MIND-Lab/OCTIS)
> 📌 Silvia Terragni, Elisabetta Fersini, Bruno Giovanni Galuzzi, Pietro Tropeano, and Antonio Candelieri. 2021. OCTIS: Comparing and Optimizing Topic models is Simple!. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 263–270, Online. Association for Computational Linguistics.

### 📌 BibTeX
```bibtex
@misc{adhya2025dtect,
      title={{DTECT}: Dynamic Topic Explorer & Context Tracker}, 
      author={Suman Adhya and Debarshi Kumar Sanyal},
      year={2025},
      eprint={2507.07910},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.07910}, 
}
