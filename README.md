# 🧠 ML Model Builder
Train and test any classification dataset with any listed models

**🔗 Deployed App:** [Launch on Hugging Face Spaces](https://huggingface.co/spaces/Dhananjaykhengare/ml_model_builder)

**🔗 Video Presentation:** [YouTube](https://www.youtube.com/watch?v=EwHgsqMNW3U)

**🔗 Download Dataset:** [Sample Dataset](https://github.com/Psuuuuu/ml_model_builder/tree/main/data)

---

## 📌 Project Overview

Since I chose the DS-2 problem, I built a UI-based tool that allows users to experiment with any classification dataset using different ML models. Users can try various combinations of features and hyperparameters to compare model performance in real time.

---

## ⚙️ Key Features & Technologies

- Upload any CSV classification dataset  
- Select features and target column  
- Handle missing values and apply normalization  
- Apply transformations like label encoding and z-score normalization  
- Train and tune models:  
  - Logistic Regression  
  - Decision Tree Classifier  
  - Random Forest Classifier  
  - Support Vector Machine (SVM)  
  - Gaussian Naive Bayes  
- Visualize metrics and compare results via dashboard and summary  

**Built With:**  
Python, Gradio, scikit-learn, pandas, numpy

---

## 🔄 Transformation Options

For each selected feature (in order), specify a transformation:  
**Allowed:** No Transformation, Label Encode, Normalize

**Example Input:**  
No Transformation, Label Encode, Normalize

---

## 🛠 Setup Instructions

1. Clone the repo:
```
git clone https://github.com/Psuuuuu/ml_model_builder.git
cd ml_model_builder
```

2. Install requirements:
```
pip install -r requirements.txt
```

3. Run the app:
```
python app.py
```

---

