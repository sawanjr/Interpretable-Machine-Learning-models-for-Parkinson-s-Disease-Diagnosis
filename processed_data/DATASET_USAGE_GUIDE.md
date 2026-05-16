# ParkinSenseDB Processed Datasets Documentation

This document describes the processed datasets generated from the ParkinSenseDB (Parkinson's Disease Gait Dataset) and provides examples of how to use them with various machine learning models.

## Dataset Overview

The ParkinSenseDB dataset contains IMU (Inertial Measurement Unit) sensor data from 53 subjects (29 PD patients, 24 healthy controls) performing gait trials. The data has been preprocessed to extract 129 statistical and frequency-domain features from the raw sensor signals.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Subjects | 53 |
| PD Subjects | 29 |
| Control Subjects | 24 |
| Total Trials | 735 |
| Number of Features | 129 |
| Statistical Features | 105 |
| Frequency Features | 24 |

---

## Dataset Files

### 1. `subject_level_features.csv`

**Description**: Features aggregated at the subject level (mean of all trials per subject)

**Use Case**: Subject-level classification where each subject is treated as one sample. Suitable for diagnostic applications where you want to classify a person as PD or Control.

**Shape**: 53 rows × 134 columns

**Key Columns**:
- `SubjectID`: Unique identifier for each subject (1-54, with some gaps)
- `Label`: Binary target variable (0 = Control, 1 = PD)
- `Sex`: Gender of the subject
- `Age`: Age of the subject
- `Status`: Text label ('PD' or 'Control')
- 129 feature columns (see Feature Description below)

**When to Use**:
- Small sample size classification (n=53)
- Person-level diagnosis
- When you want one prediction per person
- Clinical decision support systems

**Limitations**:
- Small sample size may lead to overfitting
- Subject-level aggregation loses trial-to-trial variability

### 2. `trial_level_features.csv`

**Description**: Features extracted from individual gait trials (each subject has multiple trials)

**Use Case**: Trial-level classification where each gait trial is treated as an independent sample. Suitable for analyzing gait patterns and variability within subjects.

**Shape**: 735 rows × 137 columns

**Key Columns**:
- `SubjectID`: Subject identifier
- `Test`: Test session (e.g., 'TEST A', 'TEST B')
- `Gait`: Gait trial identifier (e.g., 'gait1', 'gait2')
- `TrialLength`: Number of samples in the trial
- `Label`: Binary target variable (0 = Control, 1 = PD)
- `Sex`, `Age`, `Status`: Demographic information
- 129 feature columns

**When to Use**:
- Larger sample size (n=735)
- Analyzing gait variability
- Time-series or sequential analysis
- Understanding trial-to-trial differences
- More robust model training with more data points

**Important Considerations**:
- Trials from the same subject are NOT independent (data leakage risk)
- Use subject-wise cross-validation to avoid overfitting
- Consider clustering or hierarchical models to account for subject-level structure

---

## Feature Description

The 129 features are extracted from 9 IMU sensor channels (3 axes each for Magnetometer, Gyroscope, and Accelerometer) plus 2 magnitude features.

### Statistical Features (105 features)

For each of the 9 sensor channels:
- `_mean`: Mean value
- `_std`: Standard deviation
- `_min`: Minimum value
- `_max`: Maximum value
- `_median`: Median value
- `_q25`: 25th percentile
- `_q75`: 75th percentile
- `_range`: Range (max - min)
- `_rms`: Root mean square
- `_energy`: Signal energy (sum of squares)
- `_zero_crossings`: Number of zero crossings

Plus 3 magnitude features:
- `AccMagnitude_mean`, `AccMagnitude_std`, `AccMagnitude_max`
- `GyroMagnitude_mean`, `GyroMagnitude_std`, `GyroMagnitude_max`

### Frequency-Domain Features (24 features)

For each of the 6 channels (Gyroscope XYZ, Accelerometer XYZ):
- `_dominant_freq`: Dominant frequency from FFT
- `_dominant_amp`: Amplitude at dominant frequency
- `_spectral_energy`: Total spectral energy
- `_spectral_entropy`: Spectral entropy (signal complexity)

---

## Usage Examples

### Example 1: Subject-Level Classification with Random Forest

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load trial-level data
df = pd.read_csv('processed_data/trial_level_features.csv')

# Prepare features and target
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols]
y = df['Label']

# Split data (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Important Features:")
print(feature_importance.head(10))
```

### Example 2: Trial-Level Classification with Subject-Wise Cross-Validation

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load trial-level data
df = pd.read_csv('processed_data/trial_level_features.csv')

# Prepare features and target
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 
                               'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols]
y = df['Label']
groups = df['SubjectID']  # Important: use SubjectID as groups

# Subject-wise cross-validation
gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

accuracies = []
for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {fold + 1}: Accuracy = {acc:.3f}")

print(f"\nMean CV Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
```

### Example 3: Support Vector Machine (SVM)

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Use subject-level data for simplicity
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols]
y = df['Label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVM with RBF kernel
svm = SVC(kernel='rbf', random_state=42)

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_scaled, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")
```

### Example 4: Neural Network with PyTorch

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols].values
y = df['Label'].values

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define neural network
class PDClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PDClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model
model = PDClassifier(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f'\nTest Accuracy: {accuracy:.3f}')
```

### Example 5: XGBoost with Feature Selection

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols]
y = df['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature selection (select top 50 features)
selector = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train_selected, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test_selected)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

# Selected features
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
print(f"\nSelected Features: {selected_features}")
```

### Example 6: Logistic Regression with L1 Regularization (Feature Selection)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols]
y = df['Label']

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# L1 regularized logistic regression
lr = LogisticRegression(
    penalty='l1', 
    solver='liblinear', 
    C=0.1,  # Inverse of regularization strength
    random_state=42
)

# Cross-validation
cv_scores = cross_val_score(lr, X_scaled, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit and get selected features
lr.fit(X_scaled, y)
selected_features = [feature_cols[i] for i in np.where(lr.coef_[0] != 0)[0]]
print(f"\nSelected Features ({len(selected_features)}):")
for feat in selected_features:
    print(f"  - {feat}")
```

---

## Advanced Model Architectures

### Example 7: Tabular Transformer (TabNet)

TabNet is a deep learning architecture specifically designed for tabular data that uses attention mechanisms to select which features to use for each decision.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Note: Install pytorch-tabnet first
# pip install pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Load trial-level data
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols].values
y = df['Label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features (TabNet can handle unscaled but scaling helps)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TabNet Classifier
clf = TabNetClassifier(
    n_d=8,              # Width of decision layer
    n_a=8,              # Width of attention layer
    n_steps=3,          # Number of decision steps
    gamma=1.3,          # Relaxation factor
    lambda_sparse=1e-4, # Sparsity regularization
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=1
)

# Train
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=20,
    batch_size=16,
    virtual_batch_size=8
)

# Predict
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = clf.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)
print("\nTop 10 Features by TabNet Attention:")
print(importance_df.head(10))
```

### Example 8: Voting Ensemble (Multiple Models)

Combine predictions from multiple models using voting to improve robustness.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols]
y = df['Label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Individual models
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)

# Voting Ensemble (soft voting uses probabilities)
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('svm', svm),
        ('lr', lr),
        ('mlp', mlp),
        ('xgb', xgb)
    ],
    voting='soft'  # Use 'hard' for majority voting
)

# Evaluate with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Random Forest': rf,
    'SVM': svm,
    'Logistic Regression': lr,
    'MLP': mlp,
    'XGBoost': xgb,
    'Voting Ensemble': voting_clf
}

print("Cross-Validation Results:")
print("-" * 50)
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"{name:25s}: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Example 9: Stacking Ensemble (Meta-Learner)

Use a meta-learner to combine predictions from base models.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import RidgeClassifierCV

# Base estimators
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=4, random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
]

# Meta-learner
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=RidgeClassifierCV(),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

# Evaluate
scores = cross_val_score(stacking_clf, X_scaled, y, cv=cv, scoring='accuracy')
print(f"Stacking Ensemble CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Fit and analyze
stacking_clf.fit(X_scaled, y)
print("\nBase estimator weights from meta-learner:")
if hasattr(stacking_clf.final_estimator_, 'coef_'):
    coefs = stacking_clf.final_estimator_.coef_[0]
    for (name, _), coef in zip(estimators, coefs):
        print(f"  {name}: {coef:.3f}")
```

### Example 10: Deep Neural Network with Attention

A more sophisticated neural network with attention mechanism for feature weighting.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols].values
y = df['Label'].values

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class AttentionLayer(nn.Module):
    """Self-attention mechanism for feature weighting"""
    def __init__(self, feature_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights, attention_weights

class AdvancedPDClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.4):
        super(AdvancedPDClassifier, self).__init__()
        
        # Attention mechanism
        self.attention = AttentionLayer(input_dim, input_dim // 2)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        # Feature extractor layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Classifier
        self.classifier = nn.Linear(prev_dim, 2)
        
    def forward(self, x):
        # Apply attention
        x_attended, attention_weights = self.attention(x)
        
        # Batch normalization
        x_normalized = self.batch_norm(x_attended)
        
        # Feature extraction
        features = self.feature_extractor(x_normalized)
        
        # Classification
        output = self.classifier(features)
        
        return output, attention_weights

# Initialize model
model = AdvancedPDClassifier(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                        factor=0.5, patience=10)

# Training with early stopping
best_accuracy = 0
patience = 20
patience_counter = 0

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        test_outputs, _ = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test_tensor).float().mean().item()
    
    scheduler.step(accuracy)
    
    # Early stopping
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}, '
              f'Accuracy: {accuracy:.3f}, Best: {best_accuracy:.3f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Load best model and evaluate
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    test_outputs, attention_weights = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    final_accuracy = accuracy_score(y_test, predicted.numpy())
    print(f'\nFinal Test Accuracy: {final_accuracy:.3f}')
    
    # Get feature importance from attention
    mean_attention = attention_weights.mean(dim=0).numpy()
    top_features_idx = np.argsort(mean_attention)[-10:][::-1]
    print("\nTop 10 Features by Attention:")
    for idx in top_features_idx:
        print(f"  {feature_cols[idx]}: {mean_attention[idx]:.3f}")
```

### Example 11: Gaussian Process Classifier

Probabilistic classification with uncertainty estimates - useful for medical applications.

```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols].values
y = df['Label'].values

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use top features to reduce dimensionality for GP
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)

# Define kernel
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10)) + \
         WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

# Gaussian Process Classifier
gpc = GaussianProcessClassifier(
    kernel=kernel,
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=5,
    max_iter_predict=100,
    random_state=42
)

# Cross-validation with probability predictions
y_pred_proba = cross_val_predict(gpc, X_selected, y, cv=5, method='predict_proba')
y_pred = np.argmax(y_pred_proba, axis=1)

print(f"CV Accuracy: {accuracy_score(y, y_pred):.3f}")
print("\nPrediction Probabilities (first 5 subjects):")
for i in range(5):
    actual = 'PD' if y[i] == 1 else 'Control'
    prob_control = y_pred_proba[i][0]
    prob_pd = y_pred_proba[i][1]
    print(f"  Subject {i+1}: Control={prob_control:.3f}, PD={prob_pd:.3f} (Actual: {actual})")

# Fit final model to get uncertainty estimates
gpc.fit(X_selected, y)
print(f"\nOptimized Kernel: {gpc.kernel_}")
```

### Example 12: AutoML with TPOT

Automated machine learning for hyperparameter tuning and model selection.

```python
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols]
y = df['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TPOT AutoML
# Note: Install tpot first: pip install tpot
# This may take several minutes to hours depending on generations

tpot = TPOTClassifier(
    generations=5,              # Number of iterations
    population_size=20,         # Number of models per generation
    offspring_size=20,
    mutation_rate=0.9,
    crossover_rate=0.1,
    scoring='accuracy',
    cv=5,
    verbosity=2,
    random_state=42,
    config_dict='TPOT light',   # Use lighter config for faster results
    max_time_mins=10,           # Maximum time in minutes
    max_eval_time_mins=2        # Max time per model evaluation
)

# Run AutoML
print("Running TPOT AutoML (this may take a few minutes)...")
tpot.fit(X_train_scaled, y_train)

# Evaluate
accuracy = tpot.score(X_test_scaled, y_test)
print(f"\nBest model accuracy: {accuracy:.3f}")

# Export best pipeline
tpot.export('best_tpot_pipeline.py')
print("\nBest pipeline exported to 'best_tpot_pipeline.py'")

# Show best pipeline
print("\nBest Pipeline:")
print(tpot.fitted_pipeline_)
```

### Example 13: Contrastive Learning (Siamese Network)

Learn embeddings that distinguish between PD and Control subjects using contrastive loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SiameseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, x):
        return self.embedding(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # label: 1 if same class, 0 if different
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + 
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Load data
df = pd.read_csv('processed_data/trial_level_features.csv')
feature_cols = [col for col in df.columns 
                if col not in ['SubjectID', 'Test', 'Gait', 'TrialLength', 'Sex', 'Age', 'Status', 'Label']]
X = df[feature_cols].values
y = df['Label'].values

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create dataset
dataset = SiameseDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize
model = SiameseNetwork(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = ContrastiveLoss(margin=1.0)

# Generate pairs for training
def generate_pairs(X, y, num_pairs=1000):
    pairs = []
    labels = []
    n = len(X)
    
    for _ in range(num_pairs):
        idx1 = np.random.randint(0, n)
        idx2 = np.random.randint(0, n)
        
        pairs.append((X[idx1], X[idx2]))
        # Label: 0 if same class, 1 if different
        labels.append(0 if y[idx1] == y[idx2] else 1)
    
    return pairs, torch.FloatTensor(labels)

# Training
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    
    pairs, pair_labels = generate_pairs(X, y, num_pairs=200)
    
    for (x1, x2), label in zip(pairs, pair_labels):
        x1 = torch.FloatTensor(x1).unsqueeze(0)
        x2 = torch.FloatTensor(x2).unsqueeze(0)
        label = label.unsqueeze(0)
        
        optimizer.zero_grad()
        
        output1 = model(x1)
        output2 = model(x2)
        
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(pairs):.4f}')

# Use embeddings for classification with k-NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X)
    embeddings = model(X_tensor).numpy()

# k-NN on embeddings
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, embeddings, y, cv=5)
print(f"\nk-NN on learned embeddings - CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Compare with raw features
knn_raw = KNeighborsClassifier(n_neighbors=3)
scores_raw = cross_val_score(knn_raw, X, y, cv=5)
print(f"k-NN on raw features - CV Accuracy: {scores_raw.mean():.3f} ± {scores_raw.std():.3f}")
```

---

## Best Practices

### 1. Data Preprocessing
- **Always scale features** before using distance-based algorithms (SVM, Neural Networks, KNN)
- Random Forest and XGBoost are scale-invariant
- Handle missing values if any (check with `df.isnull().sum()`)

### 2. Cross-Validation Strategy
- **Subject-level data**: Use standard k-fold CV (e.g., 5-fold)
- **Trial-level data**: Use GroupKFold or GroupShuffleSplit with SubjectID as groups to prevent data leakage

### 3. Model Selection Guidelines

#### Traditional Models

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **Random Forest** | Handles non-linearity, feature importance, robust | Can overfit with small data | Baseline, feature analysis |
| **SVM** | Good for high-dimensional data, memory efficient | Sensitive to feature scaling | Small to medium datasets |
| **XGBoost** | State-of-the-art performance, regularization | Requires tuning | Competitive performance |
| **Neural Network** | Learns complex patterns, scalable | Requires more data, tuning | Large datasets, complex patterns |
| **Logistic Regression** | Interpretable, fast, regularization | Linear decision boundary | Baseline, feature selection |

#### Advanced Models

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **TabNet** | Feature selection via attention, handles tabular data well | Requires more data, slower training | Interpretable deep learning |
| **Voting/Stacking** | Combines strengths of multiple models, robust | More complex, longer training | Maximizing accuracy |
| **Attention DNN** | Learns feature importance, handles complex interactions | Risk of overfitting with small data | Complex pattern recognition |
| **Gaussian Process** | Uncertainty estimates, probabilistic | Computationally expensive, O(n³) | Medical applications needing confidence |
| **AutoML (TPOT)** | Automated tuning, discovers best pipeline | Very slow, resource intensive | When computational resources available |
| **Siamese Networks** | Learns similarity metrics, good for few-shot | Complex training procedure | Learning discriminative embeddings |

### 4. Handling Small Sample Size (Subject-Level)
- Use regularization (L1/L2)
- Apply feature selection
- Use stratified cross-validation
- Consider ensemble methods
- Be cautious of overfitting

### 5. Handling Class Imbalance
The dataset has 29 PD vs 24 Control subjects (reasonably balanced), but consider:
- Stratified sampling
- Class weights in models
- SMOTE for oversampling (if needed)

### 6. Feature Engineering Ideas
- Combine features (ratios, differences between axes)
- Principal Component Analysis (PCA) for dimensionality reduction
- Statistical aggregations across trials (std, min, max per subject from trial-level)

---

## Model Evaluation Metrics

For binary classification (PD vs Control):

- **Accuracy**: Overall correctness (good for balanced data)
- **Precision**: TP / (TP + FP) - reliability of positive predictions
- **Recall (Sensitivity)**: TP / (TP + FN) - ability to detect PD cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (threshold-independent)
- **Confusion Matrix**: Detailed breakdown of predictions

For medical applications, **Recall** is often prioritized to avoid missing PD cases.

---

## Troubleshooting

### Issue: Low accuracy on subject-level data
- **Cause**: Small sample size (n=53)
- **Solution**: Use regularization, feature selection, or bootstrap sampling

### Issue: Overfitting on trial-level data
- **Cause**: Trials from same subject are correlated
- **Solution**: Use GroupKFold, add subject ID as a feature, or use mixed-effects models

### Issue: Model doesn't generalize
- **Cause**: Data leakage or overfitting
- **Solution**: Ensure proper CV, check for correlated features, reduce model complexity

---

## References

- ParkinSenseDB: Parkinson's Disease Gait Dataset with IMU Sensors
- Features extracted: Statistical and frequency-domain from accelerometer, gyroscope, and magnetometer
- Preprocessing: Raw sensor data → 129 features per trial → Subject-level aggregation

For questions or issues, refer to the original dataset documentation or contact the dataset maintainers.

---

## Model Comparison and Recommendations

### Quick Start Guide

Based on your goal and available resources:

#### For Beginners
1. **Start with**: Random Forest or XGBoost
2. **Why**: Robust, good defaults, feature importance
3. **Time**: Minutes to setup and train

#### For Best Accuracy
1. **Try**: Voting Ensemble or Stacking
2. **Combine**: Random Forest + XGBoost + SVM + Neural Network
3. **Time**: Hours for training and tuning

#### For Interpretability
1. **Use**: Logistic Regression with L1 or TabNet
2. **Benefit**: Clear feature importance for clinical insights
3. **Time**: Minutes to hours depending on model

#### For Uncertainty Quantification
1. **Choose**: Gaussian Process Classifier
2. **Benefit**: Prediction probabilities for medical decision support
3. **Time**: Minutes (with reduced feature set)

#### For Research/Novel Approaches
1. **Experiment**: Siamese Networks or Contrastive Learning
2. **Benefit**: Learn embeddings, similarity metrics
3. **Time**: Hours to days for training

### Performance Benchmarks (Expected Ranges)

Based on typical results with this dataset:

| Model | Expected CV Accuracy | Training Time | Inference Time |
|-------|---------------------|---------------|----------------|
| Logistic Regression | 70-80% | Seconds | Milliseconds |
| Random Forest | 75-85% | Seconds | Milliseconds |
| SVM | 75-85% | Seconds | Milliseconds |
| XGBoost | 80-90% | Seconds | Milliseconds |
| Neural Network | 75-85% | Minutes | Milliseconds |
| TabNet | 80-90% | Minutes | Milliseconds |
| Voting Ensemble | 85-92% | Minutes | Milliseconds |
| Stacking Ensemble | 85-92% | Minutes | Milliseconds |
| Gaussian Process | 75-85% | Minutes | Seconds |
| Attention DNN | 80-90% | Minutes | Milliseconds |

*Note: Actual performance depends on data split, random seed, and hyperparameter tuning.*

### Computational Requirements

#### Lightweight (CPU only, < 1GB RAM)
- Logistic Regression
- Random Forest (small n_estimators)
- SVM (linear kernel)
- k-NN

#### Moderate (CPU, 2-4GB RAM)
- Random Forest (default)
- XGBoost
- SVM (RBF kernel)
- Neural Networks (small)
- Gaussian Process (reduced features)

#### Heavy (GPU recommended, > 8GB RAM)
- TabNet
- Large Neural Networks
- Deep ensembles
- Siamese Networks
- AutoML (TPOT)
- Full Gaussian Process

### Ensemble Strategy Decision Tree

```
What is your priority?
│
├── Maximum Accuracy
│   └── Use Stacking with diverse base models
│       └── Meta-learner: Ridge or Logistic Regression
│
├── Robustness/Stability
│   └── Use Voting Ensemble (Soft Voting)
│       └── Models: RF + XGBoost + SVM + MLP
│
├── Speed
│   └── Use Single XGBoost or Random Forest
│
└── Interpretability + Accuracy
    └── Use TabNet or Logistic Regression with feature selection
```

### Advanced Tips for Small Medical Datasets

1. **Data Augmentation**:
   - Add Gaussian noise to features (jittering)
   - Bootstrap resampling
   - SMOTE for synthetic samples (use with caution)

2. **Transfer Learning**:
   - Pre-train on larger gait datasets
   - Fine-tune on ParkinSenseDB

3. **Multi-Task Learning**:
   - Predict both PD status and age simultaneously
   - Shared representations can improve generalization

4. **Domain Adaptation**:
   - If you have data from other sensors/devices
   - Use adversarial training to align distributions

5. **Uncertainty-Aware Prediction**:
   - Use Monte Carlo Dropout in neural networks
   - Ensemble disagreement as uncertainty measure
   - Critical for clinical deployment

### Model Selection Checklist

Before finalizing your model:

- [ ] **Cross-validation**: Use appropriate strategy (standard for subject-level, group-wise for trial-level)
- [ ] **Metric selection**: Accuracy, F1, AUC, or Recall based on clinical requirements
- [ ] **Overfitting check**: Compare train vs validation performance
- [ ] **Feature importance**: Understand which features drive predictions
- [ ] **Stability**: Run multiple times with different seeds
- [ ] **Interpretability**: Can clinicians understand the model's decisions?
- [ ] **Uncertainty**: Does the model provide confidence estimates?
- [ ] **Computation**: Is inference time acceptable for real-time use?

### Recommended Pipeline for Publication

1. **Baseline**: Logistic Regression + Random Forest
2. **Compare**: Add XGBoost and SVM
3. **Advanced**: Try Neural Network or TabNet
4. **Ensemble**: Combine top 3-5 models
5. **Validate**: Use nested cross-validation for unbiased estimates
6. **Test**: Hold-out test set (never seen during development)
7. **Analyze**: Feature importance, error analysis, statistical significance

---

## Summary

This guide provides a comprehensive toolkit for analyzing the ParkinSenseDB dataset. From simple baselines to advanced deep learning architectures, choose the approach that fits your:

- **Computational resources**
- **Timeline**
- **Performance requirements**
- **Interpretability needs**

Remember: With small medical datasets (n=53), simpler models often perform as well as complex ones due to the bias-variance tradeoff. Start simple, validate rigorously, and only add complexity when justified by performance gains.

Good luck with your Parkinson's disease classification research!
