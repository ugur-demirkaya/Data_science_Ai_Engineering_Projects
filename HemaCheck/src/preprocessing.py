import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_data(data_path='../data/raw/blood_cell_anomaly_detection.csv'):
    """Load and return the blood cell dataset."""
    df = pd.read_csv(data_path)
    return df

def create_features(df):
    """Create new engineered features."""
    df = df.copy()
    
    # Shape index: combines circularity and eccentricity
    df['shape_index'] = df['circularity'] * (1 - df['eccentricity'])
    
    # Nucleus to cytoplasm ratio
    df['nucleus_cytoplasm_ratio'] = df['nucleus_area_pct'] / (100 - df['nucleus_area_pct'] + 1e-6)
    
    # Size category based on diameter
    df['size_category'] = pd.cut(df['cell_diameter_um'], 
                                  bins=[0, 8, 12, 20, 50], 
                                  labels=['Small', 'Medium', 'Large', 'VeryLarge'])
    
    # Granularity density
    df['granularity_density'] = df['granularity_score'] / (df['cell_area_px'] + 1e-6)
    
    # Color intensity variation
    df['color_variation'] = df[['mean_r', 'mean_g', 'mean_b']].std(axis=1)
    
    # Membrane irregularity
    df['membrane_irregularity'] = 1 - df['membrane_smoothness']
    
    return df

def prepare_features(df, target_col='anomaly_label'):
    """Prepare features for modeling."""
    
    # Define feature groups
    morphological = [
        'cell_diameter_um', 'nucleus_area_pct', 'chromatin_density',
        'cytoplasm_ratio', 'circularity', 'eccentricity', 
        'granularity_score', 'lobularity_score', 'membrane_smoothness',
        'cell_area_px', 'perimeter_px', 'shape_index', 
        'nucleus_cytoplasm_ratio', 'granularity_density', 'color_variation',
        'membrane_irregularity'
    ]
    
    color_features = ['mean_r', 'mean_g', 'mean_b', 'stain_intensity', 'color_variation']
    
    clinical = ['wbc_count_per_ul', 'rbc_count_millions_per_ul', 'hemoglobin_g_dl',
                'hematocrit_pct', 'platelet_count_per_ul', 'mcv_fl', 'mchc_g_dl']
    
    # Select available columns
    available_cols = []
    for col_group in [morphological, color_features, clinical]:
        for col in col_group:
            if col in df.columns:
                available_cols.append(col)
    
    # Encode categoricals - EXCLUDE leakage features
    le_dict = {}
    # cell_type and disease_category are EXCLUDED - they directly indicate anomaly
    # Only use demographic/metadata features that don't leak target info
    categorical_cols = ['patient_age_group', 'patient_sex', 'dataset_source', 
                       'staining_protocol']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
            available_cols.append(f'{col}_encoded')
    
    # Handle size_category if exists
    if 'size_category' in df.columns:
        df['size_category_encoded'] = df['size_category'].cat.codes
        available_cols.append('size_category_encoded')
    
    # Remove target-related columns
    cols_to_exclude = [target_col, 'cytodiffusion_anomaly_score', 
                      'cytodiffusion_classification_confidence', 'labeller_confidence_score']
    feature_cols = [c for c in available_cols if c not in cols_to_exclude]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y, feature_cols, le_dict

def split_and_scale(X, y, test_size=0.2, val_size=0.15, random_state=42, apply_smote=True):
    """Split data and apply scaling with optional SMOTE."""
    
    # First split: train vs temp (test+val)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size+val_size, random_state=random_state, stratify=y
    )
    
    # Second split: val vs test
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE on training data only
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE - Train: {len(X_train_scaled)}")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, scaler)

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                       scaler, output_dir='../data/processed'):
    """Save processed datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_val.npy', X_val)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_val.npy', y_val)
    np.save(f'{output_dir}/y_test.npy', y_test)
    
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    print(f"Data saved to {output_dir}")

def load_processed_data(input_dir='../data/processed'):
    """Load processed datasets."""
    X_train = np.load(f'{input_dir}/X_train.npy')
    X_val = np.load(f'{input_dir}/X_val.npy')
    X_test = np.load(f'{input_dir}/X_test.npy')
    y_train = np.load(f'{input_dir}/y_train.npy')
    y_val = np.load(f'{input_dir}/y_val.npy')
    y_test = np.load(f'{input_dir}/y_test.npy')
    
    scaler = joblib.load(f'{input_dir}/scaler.pkl')
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

if __name__ == '__main__':
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} samples")
    
    # Feature engineering
    df = create_features(df)
    print("Features engineered")
    
    # Prepare features
    X, y, feature_cols, le_dict = prepare_features(df)
    print(f"Features: {len(feature_cols)}")
    
    # Split and scale
    (X_train, X_val, X_test, 
     y_train, y_val, y_test, scaler) = split_and_scale(X, y)
    
    # Save
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    
    print("Preprocessing complete!")
