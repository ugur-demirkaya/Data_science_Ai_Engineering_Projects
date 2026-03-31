import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, precision_recall_curve)
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_random_forest(X_train, y_train, X_val, y_val, **params):
    """Train Random Forest model."""
    default_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }
    default_params.update(params)
    
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    
    print("Random Forest Results:")
    print(f"  Val Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"  Val F1: {f1_score(y_val, val_pred):.4f}")
    print(f"  Val ROC-AUC: {roc_auc_score(y_val, val_prob):.4f}")
    
    return model

def train_xgboost(X_train, y_train, X_val, y_val, **params):
    """Train XGBoost model."""
    default_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    default_params.update(params)
    
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train, 
             eval_set=[(X_val, y_val)],
             verbose=False)
    
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    
    print("XGBoost Results:")
    print(f"  Val Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"  Val F1: {f1_score(y_val, val_pred):.4f}")
    print(f"  Val ROC-AUC: {roc_auc_score(y_val, val_prob):.4f}")
    
    return model

def train_lightgbm(X_train, y_train, X_val, y_val, **params):
    """Train LightGBM model."""
    default_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    default_params.update(params)
    
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train, 
             eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
    
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    
    print("LightGBM Results:")
    print(f"  Val Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"  Val F1: {f1_score(y_val, val_pred):.4f}")
    print(f"  Val ROC-AUC: {roc_auc_score(y_val, val_prob):.4f}")
    
    return model

def train_logistic_regression(X_train, y_train, X_val, y_val, **params):
    """Train Logistic Regression model."""
    default_params = {
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': 'balanced'
    }
    default_params.update(params)
    
    model = LogisticRegression(**default_params)
    model.fit(X_train, y_train)
    
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    
    print("Logistic Regression Results:")
    print(f"  Val Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"  Val F1: {f1_score(y_val, val_pred):.4f}")
    print(f"  Val ROC-AUC: {roc_auc_score(y_val, val_prob):.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """Comprehensive model evaluation."""
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    print(f"\n=== {model_name} - TEST SET RESULTS ===")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    
    return results

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'],
               ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, weight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'],
               ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, weight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cm

def plot_roc_curves(models_dict, X_test, y_test, save_path=None):
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for (name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, 
               label=f'{name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, weight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model doesn't have feature importance")
        return
    
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp)))
    bars = ax.barh(feat_imp['feature'], feat_imp['importance'], color=colors)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, weight='bold')
    
    # Add values on bars
    for bar, val in zip(bars, feat_imp['importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return feat_imp

def save_model(model, filepath):
    """Save trained model."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load saved model."""
    return joblib.load(filepath)

if __name__ == '__main__':
    print("Model training module ready!")
