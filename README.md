# Twitter Sentiment Analysis & Prediction with GenAI Reporting

This project focuses on analyzing Twitter tweets and classifying them into three sentiment categories: **Positive, Negative, and Neutral**.  
It also includes a prediction component that applies the best-performing machine learning model to classify the sentiment of new feedback and review tweets.  
This project performs sentiment analysis on Twitter tweets, classifying them into three sentiment categories: **Positive, Negative Neutral and Irrelevant**. Beyond traditional machine learning prediction, this updated notebook integrates the power of Google Gemini GenAI to provide automated summaries of sentiment and key discussion themes, and generates comprehensive narrative reports.

The goal is to not only predict sentiment, but also to enhance the interpretability of the results for non-technical audiences through automated reporting.

---

## 📊 Dataset  

The dataset `twitter_sample.csv` contains:  

- `feedback`: The text of the tweet  
- `label`: The sentiment label (Positive, Negative, Neutral)  

---

## 🤖 Models & Performance  

| Model                        | Accuracy (%) |
|-------------------------------|--------------|
| Logistic Regression           | **67.85**    |
| Random Forest Classifier      | **78.75**    |
| Support Vector Machine (SVM)  | **70.83**    |
| Multinomial Naive Bayes       | **61.83**    |
| Bernoulli Naive Bayes         | **63.81**    |
| Gradient Boosting             | **50.91**    |

➡️ **Random Forest Classifier achieved the best performance.**

---

## 📈 Visualizations  

- [Confusion Matrix](https://github.com/Temprog/Twitter-Sentiment-Analysis-NLP/blob/main/visualizations/confusion_matrix_svm.png)
- [Sentiment Distribution (Bar Chart, Pie Chart](https://github.com/Temprog/Twitter-Sentiment-Analysis-NLP/blob/main/visualizations/sentiment_distribution.png)
- [Word Cloud of frequent terms](https://github.com/Temprog/Twitter-Sentiment-Analysis-NLP/blob/main/visualizations/wordcloud.png)
- [Model Accuracy Comparison (bar chart)](https://github.com/Temprog/Twitter-Sentiment-Analysis-NLP/blob/main/visualizations/model_accuracy.png)
- [Predicted Sentiment Distribution (New Reviews Data)](https://github.com/Temprog/Twitter-Sentiment-Analysis-NLP/blob/main/visualizations/predicted_sentiment.png)

---

## 📌 Project Workflow  

The workflow includes:  

- **Data Loading & Preprocessing**  
  - Clean text (remove URLs, mentions, hashtags, special characters, lowercase conversion)  
  - Tokenization, stopword removal, and lemmatization  
  - Handle missing values and irrelevant columns  

- **Exploratory Data Analysis (EDA)**  
  - Visualize sentiment distribution  
  - Word frequency analysis via word clouds  

- **Feature Extraction**  
  - TF-IDF Vectorization  

- **Modeling & Evaluation**  
  - Models trained: Logistic Regression, Random Forest, SVM, Multinomial Naive Bayes, Bernoulli Naive Bayes, Gradient Boosting  
  - Evaluation with accuracy, classification reports and confusion matrices  

- **Prediction on New Data**  
  - The best-performing model (Random Forest Classifier) is used to predict sentiments on new review tweets.

 **Google Gemini GenAI Integration**  
  - Connects to the Google Gemini API.
  - Generates automated narrative summaries of overall sentiment and key discussion themes based on the analysis.
  - Creates comprehensive automated narrative reports that combine project details, data analysis insights, model performance evaluation, and GenAI-generated thematic summaries.

---

**Technologies Used**
- Python
- pandas (for data manipulation and analysis)
- nltk (for natural language processing tasks like tokenization and lemmatization)
- scikit-learn (for machine learning models and evaluation)
- matplotlib and seaborn (for data visualization)
- wordcloud (for generating word clouds)
- Google Generative AI SDK (google.generativeai) for interacting with the Gemini API

---

## ⚙️ Steps to Reproduce  

1. **Google Colab** (recommended):  
   - Upload `twitter_sample.csv` and `cat.png` to Google Drive  
   - Update file paths in the notebook  
   - Run cells sequentially  

2. **Local Setup**:  
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   pip install -r requirements.txt

---

## 🌟 Importance of the Project

This project demonstrates how Natural Language Processing (NLP), machine learning and Generative AI can be applied to:

- Analyze public opinion and sentiment on social platforms at scale.
- Automate sentiment classification for large volumes of feedback.
- Generate automated summaries and narrative reports to quickly grasp key sentiment trends and discussion themes.
- Enhance the interpretability of complex data for businesses, policymakers, researchers and other stakeholders who may not have technical expertise.
- Provide actionable data-driven insights for decision-making and strategic planning.

---

## ⚠️ Limitations

- Dataset size: Small sample, may not generalize well
- Imbalanced classes: Some sentiments may dominate the dataset
- Limited feature representation: TF-IDF only; could be improved with word embeddings (Word2Vec, BERT, etc.)
- Context understanding: Traditional ML models may misinterpret sarcasm, irony, or nuanced sentiments
