# Text Summarization

This project implements an abstractive text summarization model using the CNN/DailyMail dataset. It encompasses exploratory data analysis (EDA), model training, evaluation using ROUGE metrics, and testing on both a sample from the test set and custom input data.

---

## 📚 Dataset Overview

* **Dataset**: CNN/DailyMail v3.0.0
* **Description**: An English-language dataset containing over 300,000 unique news articles from CNN and the Daily Mail, paired with human-written summaries.
* **Use Case**: Designed for training models on both extractive and abstractive summarization tasks.
* **Access**: Available via Hugging Face Datasets: [ccdv/cnn\_dailymail](https://huggingface.co/datasets/ccdv/cnn_dailymail)([Unitxt][1], [Hugging Face][2])

---

## 🧠 Model Used

The model is based on a transformer architecture suitable for sequence-to-sequence tasks, such as text summarization. It leverages pre-trained weights for efficient training and improved performance.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Saber0722/Text_Summarization.git
cd Text_Summarization
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Exploratory Data Analysis (EDA)

To understand the dataset's structure and characteristics, run the `EDA.ipynb` notebook. This will generate various plots and statistics, including:

* Distribution of article lengths
* Summary length analysis
* Word frequency distributions([Unitxt][1], [arXiv][3], [Papers with Code][4])

These insights are crucial for preprocessing and model optimization.

---

## 🚀 Model Training

Train the summarization model by executing the `train_model.ipynb` notebook.

**Important Notes**:

* **Epochs**: Set to 1 to reduce training time.
* **Hardware**: Training was performed on a GPU, which significantly reduced the training time to approximately one hour. Training on a CPU is not recommended due to the substantial time requirements.

---

## 📈 Evaluation Metrics

The model's performance is evaluated using ROUGE metrics:([Wikipedia][5])

* **ROUGE-1**: Measures the overlap of unigrams between the generated summary and the reference.
* **ROUGE-2**: Measures the overlap of bigrams.
* **ROUGE-L**: Measures the longest common subsequence, capturing sentence-level structure similarity.([Wikipedia][6])

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate automatic summarization and machine translation models by comparing the overlap between the generated and reference summaries .([Hugging Face][7])

---

## 🧪 Evaluation Approach

To minimize computational resources and time, evaluation was conducted on a single sample from the test set.

---

## 📝 Custom Data Testing

Post-training, the model was tested on custom input data to assess its generalization capabilities. This step is included at the end of the `train_model.ipynb` notebook.

---

## 🔧 NLTK Usage

The Natural Language Toolkit (NLTK) library is utilized for various preprocessing tasks, including:

* Tokenization
* Stopword removal
* Text normalization([TensorFlow][8])

These steps are essential for preparing the text data for training and evaluation.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

[1]: https://www.unitxt.ai/en/main/catalog/catalog.cards.cnn_dailymail.html?utm_source=chatgpt.com "Cnn Dailymail - Unitxt"
[2]: https://huggingface.co/datasets/ccdv/cnn_dailymail?utm_source=chatgpt.com "ccdv/cnn_dailymail · Datasets at Hugging Face"
[3]: https://arxiv.org/abs/1805.06266?utm_source=chatgpt.com "A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss"
[4]: https://paperswithcode.com/dataset/cnn-daily-mail-1?utm_source=chatgpt.com "CNN/Daily Mail Dataset - Papers With Code"
[5]: https://es.wikipedia.org/wiki/ROUGE_%28m%C3%A9trica%29?utm_source=chatgpt.com "ROUGE (métrica)"
[6]: https://en.wikipedia.org/wiki/ROUGE_%28metric%29?utm_source=chatgpt.com "ROUGE (metric)"
[7]: https://huggingface.co/spaces/evaluate-metric/rouge?utm_source=chatgpt.com "ROUGE - a Hugging Face Space by evaluate-metric"
[8]: https://www.tensorflow.org/datasets/catalog/cnn_dailymail?utm_source=chatgpt.com "cnn_dailymail | TensorFlow Datasets"
