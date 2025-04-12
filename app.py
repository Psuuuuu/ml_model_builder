import gradio as gr
import pandas as pd
import json
import os
from utils.logger import create_log_entry, log_experiment_results
from utils.file_utils import load_csv, preview_dataframe, get_column_names
from utils.training import train_models
from utils.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# For hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Needed for Naive Bayes tuning and other numeric operations
import numpy as np

# For accessing models dynamically
from utils.training import get_model_instance

# Optional: For Bayesian Optimization
try:
    from skopt import BayesSearchCV
    bayes_available = True
except ImportError:
    bayes_available = False


# Global session dictionary to hold state.
session = {
    "raw_df": None,   # Original uploaded dataframe
    "df": None,       # Processed dataframe after missing value handling, feature selection, and transformations
    "features": [],
    "target": None,
    "columns": [],
    "missing_strategy": "drop",      # Stored missing strategy
    "transformation_text": ""        # Stored transformation options (comma-separated)
}
# ---------------------------
# Dahsboard 
# ---------------------------

# ---------------------------
# Step 1: File Upload Handler
# ---------------------------
def handle_upload(file):
    if file is None:
        return "No file uploaded", None, gr.update(choices=[]), gr.update(choices=[])
    try:
        df, err = load_csv(file.name)
        session["uploaded_filename"] = file.name
        if err:
            return f"Error: {err}", None, gr.update(choices=[]), gr.update(choices=[])
        session["raw_df"] = df.copy()
        session["df"] = df.copy()  # Initialize processed df as raw df
        columns = get_column_names(df)
        session["columns"] = columns
        return (
            "File uploaded successfully!",
            preview_dataframe(df),
            gr.update(choices=columns, value=[]),
            gr.update(choices=columns, value=None)
        )
    except Exception as e:
        return f"Error: {e}", None, gr.update(choices=[]), gr.update(choices=[])

# ---------------------------
# Step 2: Global Missing Value Strategy
# ---------------------------
# def save_missing_strategy(missing_strategy):
#     raw_df = session.get("raw_df")
#     if raw_df is None:
#         return "No data available", None
#     processed_df = preprocess_data(raw_df.copy(), target_col="", missing_strategy=missing_strategy, transformation_map={})
#     session["df"] = processed_df
#     session["missing_strategy"] = missing_strategy  # Store in session
#     return f"Missing value strategy '{missing_strategy}' applied", preview_dataframe(processed_df)

def save_missing_strategy(missing_strategy):
    raw_df = session.get("raw_df")
    target_col = session.get("target", "")
    if raw_df is None:
        return "No data available", None
    processed_df = preprocess_data(raw_df.copy(), target_col=target_col, missing_strategy=missing_strategy, transformation_map={})
    session["df"] = processed_df
    session["missing_strategy"] = missing_strategy  # Store in session
    return f"Missing value strategy '{missing_strategy}' applied", preview_dataframe(processed_df)


# ---------------------------
# Step 3: Save Features and Target Selection (Filter DataFrame)
# ---------------------------
def save_feature_target_selection(features, target):
    if session.get("df") is None:
        return "No data available", "", None
    session["features"] = features
    session["target"] = target
    selected_cols = features.copy()
    if target and target not in selected_cols:
        selected_cols.append(target)
    filtered_df = session["df"][selected_cols]
    session["df"] = filtered_df
    default_trans = ", ".join(["No Transformation"] * len(features)) if features else ""
    return f"Selected {len(features)} features and target: {target}", default_trans, preview_dataframe(filtered_df)

# ---------------------------
# Step 4: Save Transformation Options
# ---------------------------
def save_transformation_options(transformation_text):
    if session.get("df") is None or not session.get("features"):
        return "No data or features available", None
    trans_list = [t.strip() for t in transformation_text.split(",")] if transformation_text.strip() != "" else []
    if len(trans_list) < len(session["features"]):
        trans_list += ["No Transformation"] * (len(session["features"]) - len(trans_list))
    transformation_mapping = {session["features"][i]: trans_list[i] for i in range(len(session["features"]))}
    df = session.get("df").copy()
    def apply_transformations(df, transformation_map):
        for col, transform in transformation_map.items():
            if transform == "Label Encode":
                if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
                    df[col] = LabelEncoder().fit_transform(df[col])
                else:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            elif transform == "Normalize":
                scaler = StandardScaler()
                df[[col]] = scaler.fit_transform(df[[col]])
        return df
    processed_df = apply_transformations(df, transformation_mapping)
    session["df"] = processed_df
    session["transformation_text"] = transformation_text  # Store in session
    return "Transformation options applied", preview_dataframe(processed_df)

# ---------------------------
# Model Training Function
# ---------------------------
def train_selected_models(experiment_title, selected_models, lr_c, lr_max_iter, dt_max_depth, dt_min_samples_split,
                          rf_n_estimators, rf_max_depth, svm_c, svm_kernel, nb_var_smoothing,
                          train_size):
    df = session.get("df")
    features = session.get("features")
    target = session.get("target")
    missing_strategy = session.get("missing_strategy", "drop")
    transformation_text = session.get("transformation_text", "")
    if df is None or not features or target is None or not selected_models:
        return "Please ensure data is uploaded, features/target selected, and models chosen."
    trans_list = [t.strip() for t in transformation_text.split(",")] if transformation_text.strip() != "" else []
    if len(trans_list) < len(features):
        trans_list += ["No Transformation"] * (len(features) - len(trans_list))
    transformation_mapping = {features[i]: trans_list[i] for i in range(len(features))}
    preprocessing_steps = [f"Missing Value: {missing_strategy}"] + [f"{k}: {v}" for k, v in transformation_mapping.items()]
    test_size = 1 - train_size
    if not set(features).issubset(df.columns):
        return "Selected features not found in the processed data."
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model_params = {
        "Logistic Regression": {"C": lr_c, "max_iter": lr_max_iter},
        "Decision Tree": {"max_depth": dt_max_depth, "min_samples_split": dt_min_samples_split},
        "Random Forest": {"n_estimators": rf_n_estimators, "max_depth": rf_max_depth},
        "SVM": {"C": svm_c, "kernel": svm_kernel},
        "Naive Bayes": {"var_smoothing": nb_var_smoothing}
    }
    training_logs = train_models(X_train, X_test, y_train, y_test, selected_models, model_params, preprocessing_steps)
    session["trained_models"] = {model: training_logs[model]["model"] for model in selected_models}
    session["X_test"] = X_test
    session["y_test"] = y_test
    experiment_logs = []
    for model_name in selected_models:
        entry = create_log_entry(
            experiment_title,
            model_name,
            model_params[model_name],
            "",
            preprocessing_steps,
            training_logs[model_name]["metrics"],
            training_logs[model_name].get("training_time", 0),
            training_logs[model_name]["model"]
        )
        experiment_logs.append(entry)
    log_experiment_results(experiment_logs)
    formatted_results = "\n".join([f"{model}: {training_logs[model]['metrics']}" for model in selected_models])
    return formatted_results

# ---------------------------
# Hyperparameter Tuning Function (Grid Search Example)
# ---------------------------
def run_hyperparameter_tuning(experiment_title, selected_models):
    df = session.get("df")
    features = session.get("features")
    target = session.get("target")

    if df is None or not features or target is None or not selected_models:
        return "Please ensure data is uploaded, features/target selected, and models chosen.", None

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    strategy_map = {
        "Grid Search": GridSearchCV,
        "Random Search": RandomizedSearchCV
    }
    if bayes_available:
        from skopt import BayesSearchCV
        strategy_map["Bayesian Optimization"] = BayesSearchCV

    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "max_iter": [100, 200, 300]},
        "Decision Tree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "Naive Bayes": {"var_smoothing": np.logspace(-10, -8, 5)}
    }

    all_logs = []
    status_texts = []

    for model_name in selected_models:
        best_overall_score = -1
        best_overall_summary = None

        for strategy_name, strategy_cls in strategy_map.items():
            try:
                model = get_model_instance(model_name, {})

                if strategy_name == "Grid Search":
                    searcher = strategy_cls(
                        model,
                        param_grid=param_grids[model_name],
                        scoring="accuracy",
                        cv=5
                    )
                elif strategy_name == "Random Search":
                    searcher = strategy_cls(
                        model,
                        param_distributions=param_grids[model_name],
                        scoring="accuracy",
                        cv=5,
                        n_iter=min(10, len(list(ParameterGrid(param_grids[model_name]))))
                    )
                elif strategy_name == "Bayesian Optimization":
                    searcher = strategy_cls(
                        model,
                        search_spaces=param_grids[model_name],
                        scoring="accuracy",
                        cv=5,
                        n_iter=10
                    )
                else:
                    continue

                searcher.fit(X_train, y_train)
                best_estimator = searcher.best_estimator_
                best_params = searcher.best_params_

                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                y_train_pred = best_estimator.predict(X_train)
                y_test_pred = best_estimator.predict(X_test)

                metrics = {
                    "accuracy_train": accuracy_score(y_train, y_train_pred),
                    "accuracy_test": accuracy_score(y_test, y_test_pred),
                    "precision_train": precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    "precision_test": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    "recall_train": recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    "recall_test": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    "f1_score_train": f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    "f1_score_test": f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                }

                log_entry = create_log_entry(
                    experiment_title,
                    f"Hyperparameter Tuned {model_name} ({strategy_name})",
                    best_params,
                    "",
                    [f"Strategy: {strategy_name}"],
                    metrics,
                    0,
                    best_estimator
                )
                all_logs.append(log_entry)

                if searcher.best_score_ > best_overall_score:
                    best_overall_score = searcher.best_score_
                    best_overall_summary = f"{model_name} ({strategy_name}):\n" + "\n".join(
                        [f"{k}: {v:.4f}" for k, v in metrics.items()]
                    )

            except Exception as e:
                continue

        if best_overall_summary:
            status_texts.append(best_overall_summary)
        else:
            status_texts.append(f"{model_name}: All tuning strategies failed.")

    log_experiment_results(all_logs)
    return "\n\n".join(status_texts), "Tuning complete!"




###--------------------dahsboard 


###--------------------dahsboard 




# ---------------------------
# Gradio Interface Layout
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## ML Model Builder")
    
    with gr.Tab("Data Upload & Preprocessing"):
        # Step 1: File Upload
        gr.Markdown("### Step 1: Upload File")
        with gr.Row():
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
        df_preview = gr.Dataframe(label="Raw Data Preview", interactive=False)
        
        # Step 2: Global Missing Value Strategy
        gr.Markdown("### Step 2: Global Missing Value Strategy")
        missing_strategy_dropdown = gr.Dropdown(
            label="Missing Value Strategy",
            choices=["drop", "mean", "median", "mode"],
            value="drop",
            info="Select how to handle missing values for all columns."
        )
        save_missing_btn = gr.Button("Save Missing Value Strategy")
        missing_status = gr.Textbox(label="Missing Strategy Status", interactive=False)
        missing_preview = gr.Dataframe(label="Data Preview after Missing Strategy", interactive=False)
        
        # Step 3: Select Features and Target
        gr.Markdown("### Step 3: Select Features and Target")
        feature_selector = gr.CheckboxGroup(label="Select Input Features", choices=[], interactive=True)
        target_selector = gr.Dropdown(label="Select Target Column", choices=[], interactive=True)
        save_features_btn = gr.Button("Save Features and Target")
        features_status = gr.Textbox(label="Features/Target Status", interactive=False)
        features_preview = gr.Dataframe(label="Data Preview after Feature Selection", interactive=False)
        
        # Step 4: Transformation Options
        gr.Markdown("### Step 4: Transformation Options")
        gr.Markdown(
            "For each selected feature (in order), specify a transformation. Allowed options: **No Transformation**, **Label Encode**, **Normalize**. "
            "Enter your choices as a comma-separated list. E.g.: No Transformation, Label Encode, Normalize"
        )
        transformation_text = gr.Textbox(label="Transformation Options", placeholder="E.g. No Transformation, Label Encode, Normalize", lines=1)
        save_transformation_btn = gr.Button("Save Transformation Options")
        transformation_status = gr.Textbox(label="Transformation Status", interactive=False)
        transformation_preview = gr.Dataframe(label="Data Preview after Transformation", interactive=False)
    
    with gr.Tab("Model Training"):
        gr.Markdown("### Model Training and Experiment Logging")
        # Global Experiment Title Input
        experiment_title_input = gr.Textbox(label="Experiment Title", placeholder="Enter a title for this experiment", lines=1)
        
        gr.Markdown("### Model Selection and Hyperparameter Tuning")
        model_selector = gr.CheckboxGroup(
            label="Select Models to Train",
            choices=["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Naive Bayes"],
            value=[], interactive=True
        )
        with gr.Column(visible=False) as lr_col:
            gr.Markdown("**Logistic Regression**")
            lr_c = gr.Slider(0.01, 10.0, step=0.01, value=1.0, label="C", interactive=True)
            lr_max_iter = gr.Slider(50, 500, step=10, value=100, label="Max Iterations", interactive=True)
        with gr.Column(visible=False) as dt_col:
            gr.Markdown("**Decision Tree**")
            dt_max_depth = gr.Slider(1, 50, step=1, value=10, label="Max Depth", interactive=True)
            dt_min_samples_split = gr.Slider(2, 10, step=1, value=2, label="Min Samples Split", interactive=True)
        with gr.Column(visible=False) as rf_col:
            gr.Markdown("**Random Forest**")
            rf_n_estimators = gr.Slider(10, 200, step=10, value=100, label="N Estimators", interactive=True)
            rf_max_depth = gr.Slider(1, 50, step=1, value=10, label="Max Depth", interactive=True)
        with gr.Column(visible=False) as svm_col:
            gr.Markdown("**SVM**")
            svm_c = gr.Slider(0.01, 10.0, step=0.01, value=1.0, label="C", interactive=True)
            svm_kernel = gr.Radio(["linear", "poly", "rbf", "sigmoid"], value="rbf", label="Kernel", interactive=True)
        with gr.Column(visible=False) as nb_col:
            gr.Markdown("**Naive Bayes**")
            nb_var_smoothing = gr.Slider(1e-10, 1e-5, step=1e-10, value=1e-9, label="Var Smoothing", interactive=True)
    
        model_columns = {
            "Logistic Regression": lr_col,
            "Decision Tree": dt_col,
            "Random Forest": rf_col,
            "SVM": svm_col,
            "Naive Bayes": nb_col,
        }
    
        def toggle_model_ui(selected_models):
            updates = []
            for model_name, panel in model_columns.items():
                updates.append(gr.update(visible=(model_name in selected_models)))
            return updates
    
        model_selector.change(
            fn=toggle_model_ui,
            inputs=model_selector,
            outputs=[lr_col, dt_col, rf_col, svm_col, nb_col]
        )
    
        gr.Markdown("### Training Parameters")
        train_slider = gr.Slider(minimum=0.5, maximum=0.9, step=0.05, value=0.8, label="Training Set Size (proportion)", interactive=True)
        train_btn = gr.Button("Train Selected Models")
        training_output = gr.Textbox(label="Training Output", lines=8, interactive=False)
    

# ---------------------------
# Define Component Interactions
# ---------------------------
    file_input.change(
        fn=handle_upload,
        inputs=file_input,
        outputs=[upload_status, df_preview, feature_selector, target_selector]
    )
    
    save_missing_btn.click(
        fn=save_missing_strategy,
        inputs=missing_strategy_dropdown,
        outputs=[missing_status, missing_preview]
    )
    
    save_features_btn.click(
        fn=save_feature_target_selection,
        inputs=[feature_selector, target_selector],
        outputs=[features_status, transformation_text, features_preview]
    )
    
    save_transformation_btn.click(
        fn=save_transformation_options,
        inputs=transformation_text,
        outputs=[transformation_status, transformation_preview]
    )
    
    train_btn.click(
        fn=train_selected_models,
        inputs=[
            experiment_title_input,
            model_selector,
            lr_c, lr_max_iter,
            dt_max_depth, dt_min_samples_split,
            rf_n_estimators, rf_max_depth,
            svm_c, svm_kernel,
            nb_var_smoothing,
            train_slider
        ],
        outputs=training_output
    )
    with gr.Tab("Hyperparameter Tuning"):
        gr.Markdown("### Fully Automatic Hyperparameter Tuning")
        gr.Markdown(
            "This step will automatically tune the selected models using **three search strategies**:\n"
            "- **Grid Search**\n"
            "- **Random Search**\n"
            "- **Bayesian Optimization** (if available)\n\n"
            "The best-performing result from each strategy will be logged, and the top strategy will be shown below."
        )
        experiment_title_hp = gr.Textbox(label="Experiment Title", placeholder="Enter experiment title")
        model_selector_hp = gr.CheckboxGroup(
            label="Select Models for Auto-Tuning",
            choices=["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Naive Bayes"],
            value=[], interactive=True
        )
        run_tune_btn = gr.Button("Run Hyperparameter Tuning")
        tuning_output = gr.Textbox(label="Tuning Output", lines=10, interactive=False)

        run_tune_btn.click(
            fn=run_hyperparameter_tuning,
            inputs=[experiment_title_hp, model_selector_hp],
            outputs=[tuning_output, gr.Textbox(visible=False)]
        )
    with gr.Tab("Dashboard"):
        log_df = gr.State(pd.DataFrame())

        def load_log_dataframe_dynamic():
            import os, json, pandas as pd

            log_path = "experiments/logs/experiment_log.jsonl"
            if not os.path.exists(log_path):
                return pd.DataFrame([{"Message": "No logs found. Train or tune a model."}])

            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            rows = []
            for line in lines:
                try:
                    row = json.loads(line)
                    metrics = row.get("metrics", {})
                    entry = {
                        "Experiment": row.get("experiment_title", ""),
                        "Timestamp": row.get("timestamp", ""),
                        "Model": row.get("model", ""),
                        "Training Time (s)": round(row.get("training_time_sec", 0), 4),
                        "Inference Time (ms)": round(metrics.get("inference_time", 0) * 1000, 4),
                        "Model Size (bytes)": row.get("model_size_bytes", ""),
                        "CPU (%)": row.get("system_info", {}).get("cpu_utilization", ""),
                        "Memory (MB)": row.get("system_info", {}).get("memory_used_mb", ""),
                        "CPU Name": row.get("system_info", {}).get("cpu", ""),
                        "Hyperparameters": json.dumps(row.get("hyperparameters", {})),
                    }
                    for k, v in metrics.items():
                        if k != "inference_time":
                            entry[k] = round(v, 4) if isinstance(v, (float, int)) else v
                    rows.append(entry)
                except Exception as e:
                    continue

            return pd.DataFrame(rows)

        refresh_button = gr.Button("🔄 Refresh Dashboard")
        dashboard_table = gr.Dataframe(
            value=load_log_dataframe_dynamic(),
            interactive=True,
            wrap=False,
            
        )

        refresh_button.click(
            fn=load_log_dataframe_dynamic,
            outputs=dashboard_table,
        )

    with gr.Tab("Summary"):

        gr.Markdown("### 🔍 Best Models by Metric")
        gr.Markdown(
            "- ✅ Automatically finds the **best model** for each evaluation metric from all logged experiments.\n"
            "- 🔁 Use the **Refresh** button to update this view after new training or tuning."
        )

        summary_df = gr.Dataframe(label="Top Models by Metric", interactive=False)

        def refresh_summary():
            import pandas as pd, os, json

            log_path = "experiments/logs/experiment_log.jsonl"
            if not os.path.exists(log_path):
                return pd.DataFrame([{"Message": "No logs found. Train or tune a model first."}])

            df = pd.read_json(log_path, lines=True)
            metric_keys = [
                "accuracy_test", "precision_test", "recall_test", "f1_score_test"
            ]

            best_rows = []

            for metric in metric_keys:
                best = None
                best_score = -float("inf")

                for _, row in df.iterrows():
                    score = row.get("metrics", {}).get(metric)
                    if isinstance(score, (int, float)) and score > best_score:
                        best = row
                        best_score = score

                if best is not None:
                    best_rows.append({
                        "Metric": metric,
                        "Best Score": round(best_score, 4),
                        "Model": best.get("model"),
                        "Experiment": best.get("experiment_title"),
                        "Timestamp": best.get("timestamp"),
                        "Hyperparameters": json.dumps(best.get("hyperparameters", {})),
                    })

            summary_df_result = pd.DataFrame(best_rows)
            if not summary_df_result.empty:
                return summary_df_result
            else:
                return pd.DataFrame([{"Message": "No valid metrics found in logs."}])

        refresh_btn = gr.Button("🔁 Refresh")
        refresh_btn.click(fn=refresh_summary, outputs=summary_df)

        # Load initial data
        summary_df.value = refresh_summary()


demo.launch()