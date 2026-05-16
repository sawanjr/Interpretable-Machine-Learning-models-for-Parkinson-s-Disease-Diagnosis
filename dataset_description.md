# Dataset Description

## ParkinSenseDB Overview

ParkinSenseDB is a multimodal gait dataset designed for Parkinson's disease research. The dataset includes inertial measurement unit (IMU) sensor data, motion-capture data, and electromyography (EMG) data collected from subjects performing various gait trials.

## Data Usage Scope

This project will exclusively utilize IMU time-series data for Parkinson versus non-Parkinson classification. Motion-capture and EMG data are considered out of scope for the initial phase.

---

## File Structure and Relationships

### Dataset Organization

The dataset consists of two main directories:

1. **Full_Dataset/** - Contains raw IMU sensor data organized by subject and test session
2. **Demography_Annotation_and_Experimental_protocol/** - Contains subject metadata, labels, and experimental documentation

### File Relationship Diagram

```
Demographic_Information.xlsx ─────┐
                                   ├──→ Subject ID ──┐
Annotation.csv ───────────────────┤                 ├──→ IMU Data Files
                                               └──→ (Full_Dataset/Subject#/TEST X/gait*/data.csv)
Experimental_protocol.pdf ────────→ Protocol Details
```

---

## Supporting Metadata Files

### 1. Demographic_Information.xlsx

**Location:** `Demography_Annotation_and_Experimental_protocol/Demography_Annotation_and_Experimental_protocol/Demographic_Information.xlsx`

**Purpose:** Contains subject demographic information and disease status labels

**Columns:**

| Column | Description | Example Values |
|--------|-------------|----------------|
| ID | Unique subject identifier (3-digit string) | 001, 002, 003, ... 054 |
| Sex | Subject biological sex | Male, Female |
| Status | Disease status classification | Control (healthy), PD (Parkinson's Disease) |
| Age | Subject age in years | 36, 41, 49, 71, 74, 76 |

**Dataset Statistics:**
- **Total Subjects:** 53
- **Parkinson's Disease (PD):** 29 subjects
- **Control (Healthy):** 23 subjects
- **Age Range:** 36 - 76 years
- **Sex Distribution:** ~28 Female, ~25 Male

**Note:** Status values are "Control" for healthy subjects and "PD" for Parkinson's disease patients. This column serves as the primary label for binary classification.

---

### 2. Annotation.csv

**Location:** `Demography_Annotation_and_Experimental_protocol/Demography_Annotation_and_Experimental_protocol/Annotation.csv`

**Purpose:** Maps gait trial recordings to subject metadata and provides frame count information

**Columns:**

| Column | Description | Example Values |
|--------|-------------|----------------|
| ID | Subject identifier (matches Demographic_Information.xlsx) | 001, 002, ... 054 |
| test | Experimental test session identifier | TEST A, TEST B, TEST C, TEST D |
| gait | Gait trial number within test session | gait1, gait2, ... gait17 |
| Frame | Total number of time frames/samples in the recording | 1979, 2358, 2745, etc. |

**Dataset Statistics:**
- **Total Records:** 747 gait trials
- **Unique Subjects:** 53
- **Trials per Subject:** 12-16 trials (varies by subject)

**Key Points:**
- Each row represents one gait trial recording
- The `Frame` column indicates the length of the time-series data in the corresponding IMU CSV file
- Multiple test sessions (A, B, C, D) per subject capture different gait conditions
- Gait trial numbering is NOT globally consistent across subjects (gait1 in TEST A for subject 001 is different from gait1 in TEST A for subject 002)

---

### 3. Experimental_protocol.pdf

**Location:** `Demography_Annotation_and_Experimental_protocol/Demography_Annotation_and_Experimental_protocol/Experimental_protocol.pdf`

**Purpose:** Documents the experimental setup, data collection procedures, and test conditions

**Contents (from scope):**
- TEST A, TEST B, TEST C, TEST D represent different experimental conditions
- Each test captures gait under specific protocols
- Motion-capture and EMG data are collected simultaneously with IMU data (out of scope for this project)

---

## IMU Sensor Data Files

### Data Location Pattern

```
Full_Dataset/Full_Dataset/{SubjectID}/{TestSession}/{gaitTrial}/data.csv
```

**Example Paths:**
- `Full_Dataset/Full_Dataset/001/TEST A/gait1/data.csv`
- `Full_Dataset/Full_Dataset/001/TEST B/gait10/data.csv`
- `Full_Dataset/Full_Dataset/054/TEST D/gait13/data.csv`

### IMU Channels (Columns)

Each IMU data CSV file contains **10 columns**:

| Column | Unit | Description |
|--------|------|-------------|
| Frame | (integer) | Sequential time frame index starting from variable offset (e.g., 637, 638, 639...) |
| MagnetometerX | mGa (milliGauss) | X-axis magnetic field measurement |
| MagnetometerY | mGa (milliGauss) | Y-axis magnetic field measurement |
| MagnetometerZ | mGa (milliGauss) | Z-axis magnetic field measurement |
| GyroscopeX | dps (degrees per second) | X-axis angular velocity (rotation rate) |
| GyroscopeY | dps (degrees per second) | Y-axis angular velocity (rotation rate) |
| GyroscopeZ | dps (degrees per second) | Z-axis angular velocity (rotation rate) |
| AccelerometerX | g (gravity) | X-axis linear acceleration |
| AccelerometerY | g (gravity) | Y-axis linear acceleration |
| AccelerometerZ | g (gravity) | Z-axis linear acceleration |

### Data Format Specifications

| Property | Value |
|----------|-------|
| File Format | CSV (Comma-Separated Values) |
| Header Row | Present (first row contains column names) |
| Row Delimiter | Newline (`\n`) |
| Column Separator | Comma (`,`) |
| Encoding | UTF-8 |
| Data Type | Floating-point numbers (except Frame which is integer) |
| Missing Values | None (complete recordings) |
| Sampling Rate | Native IMU sensor rate (not explicitly documented) |

### Sample Data Row

```
Frame,MagnetometerX(mGa),MagnetometerY(mGa),MagnetometerZ(mGa),GyroscopeX(dps),GyroscopeY(dps),GyroscopeZ(dps),AccelerometerX(g),AccelerometerY(g),AccelerometerZ(g)
637,-1327,685,685,-24.6,11.5,0.5,-0.197,0.902,0.232
```

---

## Data Relationships and Joins

### How to Link Files

1. **Subject-Level Join:**
   ```
   Annotation.csv[ID] ↔ Demographic_Information.xlsx[ID]
   ```
   - Links each gait trial to subject demographics and disease status
   - Use this to assign labels (Control/PD) to IMU data

2. **File Path Resolution:**
   ```
   Annotation.csv[ID, test, gait] → Full_Dataset/Full_Dataset/{ID}/{test}/{gait}/data.csv
   ```
   - Constructs the file path to the corresponding IMU CSV
   - Example: ID=001, test=TEST A, gait=gait1 → `Full_Dataset/Full_Dataset/001/TEST A/gait1/data.csv`

3. **Data Validation:**
   ```
   Annotation.csv[Frame] == IMU CSV row count
   ```
   - The number of rows in `data.csv` should match the `Frame` column in Annotation.csv
   - Used for data integrity verification

---

## Data Preparation for Classification

### In-Scope Activities

1. **Data Loading:** Reading IMU CSV files from the Full_Dataset directory
2. **Label Assignment:** Mapping subjects to Parkinson/non-Parkinson labels using Demographic_Information.xlsx
3. **Subject-Independent Splitting:** Splitting data at the subject level to ensure no data leakage between training and test sets
4. **Time-Series Processing:** Handling variable-length gait recordings appropriately
5. **Feature Extraction:** Computing relevant features from raw IMU signals for classification

### Out of Scope

- Processing of motion-capture data
- Processing of EMG data
- Model training and hyperparameter optimization
- Algorithm selection and comparison

---

## Dataset Statistics Summary

| Metric | Value |
|--------|-------|
| Total Subjects | 53 |
| Parkinson's Disease (PD) | 29 (54.7%) |
| Control (Healthy) | 23 (43.4%) |
| Data Quality Issues | 1 (Subject 013: "Controllo" typo) |
| Total Gait Trials | 747 |
| IMU Channels per File | 10 |
| Age Range | 36 - 76 years |
| Test Sessions | 4 (TEST A, B, C, D) |
| Trials per Subject | 12-16 |

---

## Important Notes

1. **Data Quality:** Subject 013 has a typo in Status ("Controllo" instead of "Control") - requires cleaning

2. **Sex Encoding:** Minor inconsistencies in sex labeling (" Male" vs "Male" with leading space in some records)

3. **Variable-Length Data:** Gait trials have different frame counts (typically 1500-4000 frames) - requires handling for batch processing

4. **Subject Independence:** All splits must be at the subject level to prevent data leakage

5. **IMU Relevance:** For gait analysis, Accelerometer and Gyroscope data are most relevant; Magnetometer data may be less useful depending on sensor orientation and environment
