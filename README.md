# Semantic Analyzer

A comprehensive semantic analysis system that performs sentiment analysis and relationship extraction from text data using machine learning techniques.

## 📋 Overview

This project implements a semantic analyzer that can:
- **Sentiment Analysis**: Classify text as positive or negative
- **Relationship Analysis**: Extract and analyze relationships between entities
- **Entity Recognition**: Identify different types of entities (PERSON, LOCATION, ORGANIZATION)
- **Visualization**: Create insightful visualizations of relationships and model performance

## 🚀 Features

### Core Functionality
- **Text Preprocessing**: Clean and normalize text data
- **Feature Extraction**: Convert text to TF-IDF features
- **Machine Learning Models**: 
  - Naive Bayes Classifier
  - Logistic Regression
- **Model Evaluation**: Comprehensive performance metrics
- **Interactive Testing**: Test custom sentences

### Relationship Analysis
- **Relationship Type Analysis**: Identify most common relationship types
- **Entity Type Relationships**: Analyze connections between different entity types
- **Network Analysis**: Visualize entity relationship networks
- **Frequency Analysis**: Count relationship occurrences

### Visualization
- **Confusion Matrix**: Model performance visualization
- **Accuracy Comparison**: Bar charts comparing model performance
- **Relationship Charts**: Visualize relationship frequencies
- **Entity Network**: Network graphs of entity connections

## 📁 Project Structure

```
SemanticAnalyzer/
├── RelationshipAnalyzer.ipynb    # Main Jupyter notebook
├── train.json                    # Training dataset
├── test.json                     # Test dataset
├── valid.json                    # Validation dataset
└── README.md                     # This file
```

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required Python packages (see below)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd SemanticAnalyzer
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn textblob jupyter
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open the notebook**
   - Open `RelationshipAnalyzer.ipynb`
   - Run cells step by step

## 📊 Dataset Format

The project uses JSON files with the following structure:

```json
{
  "sentText": "Sentence text here",
  "relationMentions": [
    {
      "em1Text": "Entity1",
      "em2Text": "Entity2", 
      "label": "/relationship/type"
    }
  ],
  "entityMentions": [
    {
      "start": 0,
      "label": "PERSON",
      "text": "Entity Name"
    }
  ]
}
```

## 🔧 Usage

### Step-by-Step Execution

1. **Import Libraries** - Load all required Python packages
2. **Load Dataset** - Read and explore the JSON data files
3. **Data Preprocessing** - Clean text and extract features
4. **Text Vectorization** - Convert text to TF-IDF features
5. **Model Training** - Train Naive Bayes and Logistic Regression models
6. **Model Evaluation** - Compare model performance
7. **Visualization** - Create confusion matrices and comparison charts
8. **Relationship Analysis** - Analyze entity relationships
9. **Interactive Testing** - Test with custom sentences

### Testing Custom Sentences

```python
# Example usage
test_sentence = "I love this amazing product!"
prediction, confidence = predict_sentiment(test_sentence, best_model, tfidf_vectorizer)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Confidence: {confidence:.4f}")
```

## 📈 Model Performance

The system typically achieves:
- **Accuracy**: 70-85% depending on data quality
- **Precision**: Good performance on balanced datasets
- **Recall**: Effective at identifying both positive and negative sentiments

## 🎯 Key Components

### Sentiment Analysis
- Uses relationship types to determine sentiment
- Falls back to TextBlob for sentences without relationships
- Supports both positive and negative classification

### Relationship Extraction
- Analyzes `/relationship/type` patterns in data
- Categorizes relationships as positive/negative
- Extracts entity pairs and their connections

### Feature Engineering
- TF-IDF vectorization with 5000 features
- Bigram features for better context
- English stop words removal
- Text normalization and cleaning

## 🔍 Analysis Capabilities

### Relationship Types Analyzed
- **Location-based**: `/location/location/contains`
- **Person-based**: `/people/person/place_of_birth`
- **Business**: `/business/person/company`
- **Family**: `/people/person/children`
- **And many more...**

### Entity Types Recognized
- **PERSON**: People names
- **LOCATION**: Places, cities, countries
- **ORGANIZATION**: Companies, institutions
- **MISC**: Other entities

## 📊 Output Examples

### Model Performance
```
Logistic Regression Results:
Accuracy: 0.8234

Classification Report:
              precision    recall  f1-score   support
    Negative       0.81      0.78      0.79       245
    Positive       0.83      0.86      0.84       312
```

### Relationship Analysis
```
Top Relationship Types:
/location/location/contains: 1250
/people/person/place_of_birth: 890
/business/person/company: 456
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all required packages are installed
2. **Memory Issues**: Reduce `max_features` in TF-IDF vectorizer
3. **Slow Training**: Use smaller datasets for testing

### Getting Help

- Check the Jupyter notebook for detailed comments
- Ensure your virtual environment is activated
- Verify dataset format matches expected structure

## 🎉 Acknowledgments

- Built with scikit-learn for machine learning
- Uses TextBlob for additional sentiment analysis
- Visualization powered by matplotlib and seaborn

---

**Happy Analyzing! 🚀** 