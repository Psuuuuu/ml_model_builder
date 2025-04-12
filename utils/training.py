# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import time
# import joblib
# import os

# def get_model_instance(model_name, params):
#     if model_name == "Logistic Regression":
#         return LogisticRegression(**params)
#     elif model_name == "Decision Tree":
#         return DecisionTreeClassifier(**params)
#     elif model_name == "Random Forest":
#         return RandomForestClassifier(**params)
#     elif model_name == "SVM":
#         return SVC(**params)
#     elif model_name == "Naive Bayes":
#         return GaussianNB(**params)
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

# def train_models(X_train, X_test, y_train, y_test, selected_models, model_params, preprocessing_steps, experiment_name="DefaultExperiment", dataset_name="Uploaded CSV"):
#     results = {}
#     for model_name in selected_models:
#         params = model_params.get(model_name, {})
#         model = get_model_instance(model_name, params)
        
#         # ✅ Measure training time
#         start_time = time.time()
#         model.fit(X_train, y_train)
#         training_time = round(time.time() - start_time, 4)

#         # ✅ Measure model size
#         temp_model_path = f"models/{model_name.replace(' ', '_')}_temp.joblib"
#         os.makedirs("models", exist_ok=True)
#         joblib.dump(model, temp_model_path)
#         model_size = os.path.getsize(temp_model_path)
#         os.remove(temp_model_path)

#         # ✅ Predictions
#         y_train_pred = model.predict(X_train)
#         y_test_pred = model.predict(X_test)

#         # ✅ Inference time on a single sample
#         single_sample = X_test.iloc[[0]] if hasattr(X_test, "iloc") else X_test[0].reshape(1, -1)
#         start_inf = time.time()
#         _ = model.predict(single_sample)
#         inference_time = round(time.time() - start_inf, 6)

#         # ✅ Evaluation metrics
#         metrics = {
#             "accuracy_train": accuracy_score(y_train, y_train_pred),
#             "accuracy_test": accuracy_score(y_test, y_test_pred),
#             "precision_train": precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
#             "precision_test": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
#             "recall_train": recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
#             "recall_test": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
#             "f1_score_train": f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
#             "f1_score_test": f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
#             "inference_time": inference_time
#         }

#         results[model_name] = {
#             "model": model,
#             "metrics": metrics,
#             "training_time": training_time,
#             "inference_time": inference_time,
#             "model_size": model_size
#         }
#     return results


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time import perf_counter as timer  # ✅ Use perf_counter for better precision
import joblib
import os

def get_model_instance(model_name, params):
    if model_name == "Logistic Regression":
        return LogisticRegression(**params)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(**params)
    elif model_name == "Random Forest":
        return RandomForestClassifier(**params)
    elif model_name == "SVM":
        return SVC(**params)
    elif model_name == "Naive Bayes":
        return GaussianNB(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def train_models(X_train, X_test, y_train, y_test, selected_models, model_params, preprocessing_steps, experiment_name="DefaultExperiment", dataset_name="Uploaded CSV"):
    results = {}
    for model_name in selected_models:
        params = model_params.get(model_name, {})
        model = get_model_instance(model_name, params)

        # ✅ Measure training time with high precision
        start_time = timer()
        model.fit(X_train, y_train)
        training_time = round(timer() - start_time, 4)

        # ✅ Measure model size
        temp_model_path = f"models/{model_name.replace(' ', '_')}_temp.joblib"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, temp_model_path)
        model_size = os.path.getsize(temp_model_path)
        os.remove(temp_model_path)

        # ✅ Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # ✅ Inference time on a single sample
        single_sample = X_test.iloc[[0]] if hasattr(X_test, "iloc") else X_test[0].reshape(1, -1)
        start_inf = timer()
        _ = model.predict(single_sample)
        inference_time = round(timer() - start_inf, 6)

        # ✅ Evaluation metrics
        metrics = {
            "accuracy_train": accuracy_score(y_train, y_train_pred),
            "accuracy_test": accuracy_score(y_test, y_test_pred),
            "precision_train": precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            "precision_test": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "recall_train": recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
            "recall_test": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "f1_score_train": f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
            "f1_score_test": f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "inference_time": inference_time
        }

        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "training_time": training_time,
            "inference_time": inference_time,
            "model_size": model_size
        }
    return results
