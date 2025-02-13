# mlp-emotion-classifier
A Multilayer Perceptron (MLP) model built from scratch (without external machine learning libraries) to classify tweet sentiments into four categories: positive, negative, neutral, and irrelevant. This project explores fundamental neural network principles and highlights the challenges of text-based sentiment classification.


### **Project Overview**
- **Dataset**: Kaggle’s [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis?resource=download)  
- **ML Model**: Custom-built Multilayer Perceptron (MLP) using **no machine learning frameworks** (i.e., no TensorFlow, PyTorch, etc.)
- **Tech Stack**: Python, NumPy, Pandas
- **Key Features**:
  - **Tokenization & Vectorization** using TF-IDF
  - **Backpropagation** implemented manually
  - **Softmax activation** for multi-class classification
  - **Custom loss function & optimization** (SGD)
  - **Validation & Performance Evaluation**

---

### **How It Works**
1. **Data Preprocessing**:
   - Text cleaning (removal of special characters)
   - Tokenization & vectorization using **TF-IDF**
   - Sentiment labels encoded into numerical values

2. **Model Architecture**:
   - **Input Layer**: Accepts vectorized tweet data
   - **Hidden Layers**: 2 fully connected layers with **ReLU activation**
   - **Output Layer**: Softmax activation for 4 sentiment categories

3. **Training & Evaluation**:
   - **Backpropagation** for weight adjustments
   - **Categorical Cross-Entropy loss** calculation
   - **Performance Metrics**: Accuracy and validation loss

---

### **Performance**
- **Training Accuracy**: Limited generalization due to dataset imbalances.
- **Validation Accuracy**: ~26.8%  
- **Challenges**: The model struggled with class imbalance and lacked sophisticated feature representations.

#### **Potential Improvements**
- **Use Pre-trained Embeddings** (e.g., Word2Vec, GloVe, or BERT)
- **Increase Model Complexity** (Larger network, dropout layers)
- **Data Augmentation** (Handling class imbalance through oversampling)

---

### **Installation & Usage**
#### **1. Clone the Repository**
```bash
git clone https://github.com/paintbrushamster/mlp-emotion-classifier.git
cd mlp-emotion-classifier
```

#### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **3. Run the Model**
```bash
python main.py
```

- **If a trained model exists**, it loads automatically.
- **Otherwise, it trains a new model** and saves the parameters.

---

### **File Structure**
```
/mlp-emotion-classifier
├── data/
│   ├── twitter_training.csv
│   ├── twitter_validation.csv
├── src/
│   ├── main.py  
│   ├── training_model.py  
│   ├── validation.py  
│   ├── preprocess.py  # Data preprocessing
│   ├── model_definition.py  
├── experiments/
│   ├── train  # Saved model
├── notebooks/
│   ├── analysis.ipynb  # Jupyter Notebook for experimentation
├── README.md  # Project documentation

/mlp-emotion-classifier
├── /data
│   ├── twitter_training.csv
│   ├── twitter_validation.csv
├── /src
│   ├── main.py  # Main entry point
│   ├── model_definition.py # Neural network layers
│   ├── training_model.py  # Model training script
│   ├── training_df.py
│   ├── validation.py # Model validation script
│   ├── validation_df.py
├── /experiments
│   ├── training-draft.py
├── /models
│   ├── model_parameters.pkl
├── .gitignore
├── README.md

```
