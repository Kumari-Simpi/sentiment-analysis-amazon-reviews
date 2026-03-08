cat > README.md << 'EOF'
# 🛍️ Sentiment Analysis on Amazon Product Reviews

## 🚀 Live Demo
👉 [Click here to try the app](https://8501-01kk22wckqxn1yq5zr9ppey3fj.cloudspaces.litng.ai)

## 📌 Project Overview
An end-to-end NLP project that classifies Amazon product reviews as **Positive**, **Neutral**, or **Negative** using Machine Learning.

## 📂 Project Structure
\`\`\`
├── app.py                  # Streamlit web app
├── Untitled.ipynb          # Main notebook (full pipeline)
├── saved_models/           # Trained model & vectorizers
├── train_data.csv          # Training data
├── test_data.csv           # Test data
├── test_data_hidden.csv    # Hidden test data
├── all_predictions.csv     # All model predictions
└── final_submission.csv    # Best model predictions
\`\`\`

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Scikit-learn, XGBoost, NLTK, TextBlob, Pandas, NumPy, Matplotlib, Seaborn
- **Deployment:** Streamlit + Lightning AI

## 🤖 Models Used
- Logistic Regression
- Naive Bayes
- Random Forest
- XGBoost
- SVM (Linear)
- Voting Classifier
- Stacking Classifier

## ⚙️ Pipeline
1. Data Cleaning & Preprocessing
2. Text Cleaning & Lemmatization
3. Feature Engineering (TF-IDF, TextBlob Polarity)
4. Class Imbalance Handling (Oversampling & Undersampling)
5. Model Training & Evaluation
6. Unsupervised Learning (K-Means, LDA)
7. Deployment via Streamlit

## 📊 Evaluation Metrics
- Accuracy
- F1 Score (Weighted)
- ROC-AUC Score
- Confusion Matrix

## 🔧 How to Run Locally
\`\`\`bash
pip install streamlit scikit-learn xgboost nltk textblob wordcloud pandas numpy matplotlib seaborn joblib
streamlit run app.py
\`\`\`
EOF

git add README.md
git commit -m "Add live app link and complete README"
git push
