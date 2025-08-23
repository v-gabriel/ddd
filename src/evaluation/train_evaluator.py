import os
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, \
    precision_recall_curve, roc_auc_score
import logging

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def get_final_performance_report(
    all_cv_results: list,
    final_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    optimization_strategy: str = 'OptimalAccuracy'
) -> dict:
    """
    Calculates the final model performance on a hold-out test set using a single,
    robust threshold derived from the mean of optimal thresholds from CV.
    """
    # Collect optimal thresholds for the chosen strategy from CV results.
    optimal_thresholds = []
    for result in all_cv_results:
        # A model name is expected to look like 'LSTM_Fold_1_OptimalF1'.
        parts = result['Model'].split('_')

        result_model_name = parts[0]
        result_strategy = parts[-1]

        if result_model_name.lower() == model_name.lower() and result_strategy == optimization_strategy:
            optimal_thresholds.append(result['Threshold'])

    if not optimal_thresholds:
        logger.error(f"Could not find any results for strategy '{optimization_strategy}' and model '{model_name}'.")
        return None

    final_threshold = np.mean(optimal_thresholds)
    logger.info(
        f"Calculated final robust threshold for '{optimization_strategy}' for model '{model_name}' is {final_threshold:.4f} "
        f"from {len(optimal_thresholds)} folds."
    )

    if hasattr(final_model, 'predict_proba'):
        y_pred_proba_test = final_model.predict_proba(X_test)[:, 1]
        y_pred_test_binary = (y_pred_proba_test >= final_threshold).astype(int)
    else:
        # If model doesn't support probabilities, use direct predictions. Threshold is ignored.
        logger.warning(f"Model {model_name} does not have a 'predict_proba' method. "
                       f"Using direct .predict() for final evaluation. The threshold will not be applied.")

        y_pred_raw_probs = final_model.predict(X_test)
        y_pred_test_binary = np.argmax(y_pred_raw_probs, axis=1)

        final_threshold = "N/A" # Indicate that threshold was not used.

    final_metrics = calculate_metrics(y_test, y_pred_test_binary)

    report_name = f"{model_name}_Final_Report_{optimization_strategy}"
    final_report = {
        'Model': report_name,
        'Final_Threshold': final_threshold,
        'Test_Set_Accuracy': round(final_metrics.get('accuracy', 0), 4),
        'Test_Set_F1_Score': round(final_metrics.get('f1_score', 0), 4),
        'Test_Set_Precision': round(final_metrics.get('precision', 0), 4),
        'Test_Set_Recall': round(final_metrics.get('recall', 0), 4),
        'Test_Set_Specificity': round(final_metrics.get('specificity', 0), 4),
    }

    try:
        cm = confusion_matrix(y_test, y_pred_test_binary)
        logger.info(f"{report_name} - Final Test Set Confusion Matrix:\n{cm}")
    except Exception as e:
        logger.warning(f"Could not log final confusion matrix for {report_name}: {e}")

    logger.info(f"--- FINAL PERFORMANCE REPORT ({model_name}) ---")
    logger.info(final_report)
    return final_report


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate metrics"""
    try:
        drowsy_labels = [1]
        y_true_binary = np.isin(y_true, drowsy_labels).astype(int)
        y_pred_binary = np.isin(y_pred, drowsy_labels).astype(int)

        # 'weighted' average is suitable for general performance overview
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
        precision = precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)

    except Exception as e:
        logger.error(f"Error calculating basic metrics: {e}")
        accuracy = f1 = precision = recall = 0.0

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
    }

    try:
        specificity = calculate_specificity(y_true, y_pred)
        metrics['specificity'] = specificity

    except Exception as e:
        logger.error(f"Error calculating additional metrics: {e}")
        metrics.update({
            'false_alarm_rate': 0.0,
            'temporal_consistency': 0.0,
            'specificity': 0.0
        })

    return metrics


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity (true negative rate)"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificities = []
            for i in range(cm.shape[0]):
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(spec)
            specificity = np.mean(specificities)

        return specificity
    except Exception as e:
        logger.error(f"Error calculating specificity: {e}")
        return 0.0


def evaluate_fold_with_threshold_optimization(y_true, y_pred_proba, label_encoder, model_name, config):
    try:
        def get_results_dict(metrics, y_pred, threshold, optimization_identifier):
            model_name_optimized = f"{model_name}_{optimization_identifier}"

            try:
                cm = confusion_matrix(y_true, y_pred)
                logger.info(f"{model_name_optimized} - Confusion Matrix:\n{cm}")
            except Exception as e:
                logger.warning(f"Could not log confusion matrix for {model_name_optimized}: {e}")

            return {
                'Model': model_name_optimized,
                'Threshold': round(threshold, 3),
                'Accuracy': round(metrics['accuracy'], 4),
                'F1_Score': round(metrics['f1_score'], 4),
                'Precision': round(metrics['precision'], 4),
                'Recall': round(metrics['recall'], 4),
                'Specificity': round(metrics['specificity'], 4),
            }

        if y_pred_proba is None or len(np.unique(y_true)) < 2:
            logger.warning(f"Cannot perform threshold optimization for {model_name}: insufficient data")
            y_pred_default = (y_pred_proba >= 0.5).astype(int) if y_pred_proba is not None else np.zeros_like(y_true)
            return [evaluate_fold(y_true, y_pred_default, label_encoder, model_name)]

        _plot_and_save_curves(y_true, y_pred_proba, model_name, config)

        thresholds = np.linspace(0, 1, num=100)
        y_preds = (y_pred_proba[:, None] >= thresholds)  # shape: (n_samples, 100)

        all_metrics = []
        for i, thr in enumerate(thresholds):
            m = calculate_metrics(y_true, y_preds[:, i])
            m["threshold"] = thr
            all_metrics.append(m)

        # thresholds = np.unique(np.concatenate([
        #     np.array([0.0, 1.0]),
        #     np.sort(np.unique(y_pred_proba))
        # ]))
        # all_metrics = []
        # for thr in thresholds:
        #     y_pred = (y_pred_proba >= thr).astype(int)
        #     m = calculate_metrics(y_true, y_pred)
        #     m["threshold"] = thr
        #     all_metrics.append(m)

        accs = np.array([m["accuracy"] for m in all_metrics])
        f1s = np.array([m["f1_score"] for m in all_metrics])
        recalls = np.array([m["recall"] for m in all_metrics])

        idx_acc = np.argmax(accs)
        idx_f1 = np.argmax(f1s)
        recall_target = 0.95
        recall_ok = np.where(recalls >= recall_target)[0]
        idx_rec = recall_ok[np.argmax(f1s[recall_ok])] if len(recall_ok) else np.argmax(recalls)

        balanced_scores = np.array([(m['recall'] + m['specificity']) / 2 for m in all_metrics])
        idx_bal = np.argmax(balanced_scores)

        results = []
        for idx, name in zip([idx_f1, idx_acc, idx_rec, idx_bal],
                             ["OptimalF1", "OptimalAccuracy", "OptimalRecall", "Balanced"]):
            m = all_metrics[idx]
            y_pred = (y_pred_proba >= m["threshold"]).astype(int)
            metrics = calculate_metrics(y_true, y_pred)
            results.append(get_results_dict(metrics, y_pred, m["threshold"], name))

        return results
    except Exception as e:
        logger.error(f"Critical error in threshold optimization for {model_name}: {e}")
        y_pred_default = (y_pred_proba >= 0.5).astype(int) if y_pred_proba is not None else np.zeros_like(y_true)
        return [evaluate_fold(y_true, y_pred_default, label_encoder, model_name)]


# TODO: MCC?
def evaluate_fold(y_true, y_pred, label_encoder, model_name):
    """Standard fold evaluation"""
    try:
        metrics = calculate_metrics(y_true, y_pred)

        result = {
            'Model': model_name,
            'Threshold': 0.5,
            'Accuracy': round(metrics['accuracy'], 4),
            'F1_Score': round(metrics['f1_score'], 4),
            'Precision': round(metrics['precision'], 4),
            'Recall': round(metrics['recall'], 4),
            'Specificity': round(metrics['specificity'], 4),
        }

        if 'detection_latency' in metrics:
            result['Detection_Latency'] = round(metrics['detection_latency'], 4)

        try:
            cm = confusion_matrix(y_true, y_pred)
            logger.info(f"{model_name} - Confusion Matrix:\n{cm}")
        except Exception as e:
            logger.warning(f"Could not log confusion matrix for {model_name}: {e}")

        return result

    except Exception as e:
        logger.error(f"Error evaluating fold for {model_name}: {e}")
        return {
            'Model': model_name,
            'Threshold': 0.5,
            'Accuracy': 0.0,
            'F1_Score': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'Specificity': 0.0,
        }


def aggregate_cv_results(fold_results):
    """Aggregate cross-validation results with comprehensive statistics"""
    if not fold_results:
        return []

    try:
        # Group results by model base name (removing fold numbers and threshold types)
        models = {}
        for result in fold_results:
            model_full_name = result['Model']
            # extract base model name (remove _Fold_X, _OptimalF1, _OptimalRecall, _Balanced)
            base_name = \
            model_full_name.split('_Fold_')[0].split('_OptimalF1')[0].split('_OptimalRecall')[0].split('_Balanced')[0]

            if base_name not in models:
                models[base_name] = {}

            # group by threshold type
            if '_OptimalF1' in model_full_name:
                threshold_type = 'OptimalF1'
            elif '_OptimalRecall' in model_full_name:
                threshold_type = 'OptimalRecall'
            elif '_Balanced' in model_full_name:
                threshold_type = 'Balanced'
            elif '_OptimalAccuracy' in model_full_name:
                threshold_type = 'OptimalAccuracy'
            else:
                threshold_type = 'Default'

            if threshold_type not in models[base_name]:
                models[base_name][threshold_type] = []

            models[base_name][threshold_type].append(result)

        aggregated = []

        for model_name, threshold_groups in models.items():
            for threshold_type, results in threshold_groups.items():
                if len(results) < 2:
                    logger.warning(
                        f"Skipping aggregation for {model_name} [{threshold_type}] due to insufficient folds ({len(results)})")
                    continue

                metrics = [
                    'Accuracy',
                    'F1_Score',
                    'Precision',
                    'Recall',
                    'Specificity',
                ]

                suffix = f"_{threshold_type}" if threshold_type != 'Default' else ""
                agg_result = {'Model': f"{model_name}_CV_Mean{suffix}"}

                for metric in metrics:
                    values = [r[metric] for r in results if metric in r and r[metric] is not None]
                    if values:
                        agg_result[metric] = round(np.mean(values), 4)
                        agg_result[f"{metric}_Std"] = round(np.std(values), 4)
                        agg_result[f"{metric}_Min"] = round(np.min(values), 4)
                        agg_result[f"{metric}_Max"] = round(np.max(values), 4)

                aggregated.append(agg_result)

        return aggregated

    except Exception as e:
        logger.error(f"Error aggregating CV results: {e}")
        return []

def _plot_and_save_curves(y_true, y_pred_proba, model_name, config: ExperimentConfig):
    """ Helper function to generate and save ROC and PR curves."""
    try:
        plot_dir = os.path.join(config.RESULTS_EXCEL_FILE, config.SUITE_NAME, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        roc_path = os.path.join(plot_dir, f"{config.NAME}_{model_name}_ROC.png")
        plt.savefig(roc_path)
        plt.close()

        # --- Precision-Recall Curve ---
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {model_name}')
        plt.grid(True)
        pr_path = os.path.join(plot_dir, f"{config.NAME}_{model_name}_PR.png")
        plt.savefig(pr_path)
        plt.close()

        logger.info(f"Saved ROC and PR curves for {model_name} to {plot_dir}")
    except Exception as e:
        logger.error(f"Failed to generate plots for {model_name}: {e}")