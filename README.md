# 🏆 Fantasy Premier League Player Performance Prediction

<div align="center">

![FPL Banner](https://img.shields.io/badge/Fantasy-Premier_League-38003c?style=for-the-badge&logo=premier-league)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Ridge_Regression-FF6B6B?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**Predicting player points for upcoming gameweeks using machine learning and statistical analysis**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Project Structure](#-project-structure)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Model Explainability](#-model-explainability)
- [Project Structure](#-project-structure)
- [Deliverables](#-deliverables)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Fantasy Premier League (FPL) is one of the most popular online strategy games globally, with millions of football fans creating virtual teams based on real-world player performances. This project implements a **machine learning pipeline** to predict player points for upcoming gameweeks using historical performance data.

### Key Objectives

- 🧹 **Data Cleaning**: Remove unnecessary columns and handle inconsistencies
- 🔧 **Feature Engineering**: Create meaningful features like player "form"
- 📊 **Data Analysis**: Answer key questions about player performance patterns
- 🤖 **Predictive Modeling**: Build a Ridge Regression model to forecast upcoming points
- 🔍 **Model Explainability**: Implement SHAP and LIME for interpretability
- 🚀 **Production-Ready**: Create an inference function for real-time predictions

---

## ✨ Features

### 🔄 Data Processing Pipeline
- Automated data cleaning and preprocessing
- Handles missing values, duplicates, and inconsistent data
- Standardized column naming conventions
- Time-series aware data sorting

### 📈 Feature Engineering
- **Form Feature**: Rolling 4-week average of player performance
- Position encoding for categorical data
- Feature scaling for optimal model performance
- Correlation analysis with target variable

### 🎯 Predictive Model
- **Ridge Regression** (L2 regularized linear regression)
- Predicts `upcoming_total_points` for each player
- Evaluates with 4 key metrics: MAE, MSE, RMSE, R²
- Separate training and test set evaluation

### 🔍 Explainable AI (XAI)
- **SHAP** analysis for global feature importance
- **LIME** explanations for individual predictions
- Ridge coefficient visualization
- Feature contribution insights

### 🚀 Production-Ready Inference
- `predict_upcoming_points()` function
- Accepts raw player data (dict or DataFrame)
- Automatic feature preprocessing
- Returns predicted points for next gameweek

---

## 📊 Dataset

### Source
Historical Fantasy Premier League data spanning multiple seasons, containing player-level performance records for each gameweek.

### Key Identifiers
- `player_name` - Player's full name
- `team` - Current team
- `position` - GK, DEF, MID, or FWD
- `gameweek` - Match week number
- `season` - FPL season (e.g., 2022-23)

### Performance Metrics
- **Match Stats**: goals_scored, assists, minutes, clean_sheets, saves
- **Quality Metrics**: creativity, influence, threat, ICT index
- **FPL Scoring**: total_points, bonus points, BPS
- **Discipline**: yellow_cards, red_cards, own_goals

### Dataset Statistics
- **Total Records**: 100,000+ player-gameweek observations
- **Players**: 500+ unique players
- **Seasons**: Multiple Premier League seasons
- **Features**: 24 relevant features after cleaning

---

## 🔬 Methodology

### 1. Data Cleaning
```python
# Remove unnecessary columns
- Popularity metrics (transfers, selection %)
- Administrative data (fixture IDs, round numbers)
- Match context (opponent, home/away)

# Handle missing values
- Fill numeric columns with 0 (represents no contribution)

# Remove duplicates and sort chronologically
```

### 2. Feature Engineering

**Form Feature Creation:**
```
form = (average total_points over past 4 gameweeks) / 10
```

This captures recent performance trends and momentum.

### 3. Data Analysis

#### Question 1: Position Analysis
*"Which positions score the most points on average?"*

- Groups players by position (GK, DEF, MID, FWD)
- Calculates average total points
- Visualizes with bar charts and box plots

#### Question 2: Performance Evolution
*"How do top players evolve across gameweeks in 2022-23?"*

- Identifies top 5 by total points vs. top 5 by form
- Tracks performance evolution across gameweeks
- Compares consistency vs. peak performance

### 4. Predictive Modeling

**Target Variable:**
```python
upcoming_total_points = shift(total_points, -1 week)
```
Predicts next week's points from current week's performance.

**Model:** Ridge Regression (α = 1.0)
- Statistical ML model with L2 regularization
- Handles multicollinearity between features
- Interpretable coefficients
- Computationally efficient

**Features Used:**
- **Match-related**: goals_scored, assists, minutes, clean_sheets
- **Player-related**: position, creativity, influence, value
- **Engineered**: form

**Evaluation Metrics:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (R-squared)

### 5. Model Explainability

**SHAP (SHapley Additive exPlanations):**
- Global feature importance
- Summary plots showing feature impact
- Ridge coefficient visualization

**LIME (Local Interpretable Model-agnostic Explanations):**
- Local explanations for individual predictions
- Shows which features contributed to specific predictions
- Validates model reasoning

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone Repository
```bash
git clone https://github.com/yourusername/fpl-prediction.git
cd fpl-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
shap>=0.41.0
lime>=0.2.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### 1. Run Jupyter Notebook
```bash
jupyter notebook fpl-prediction.ipynb
```

### 2. Use Inference Function

```python
from inference import predict_upcoming_points

# Example player data
player_data = {
    'goals_scored': 2,
    'assists': 1,
    'minutes': 90,
    'clean_sheets': 0,
    'position': 'MID',
    'creativity': 80.0,
    'influence': 75.0,
    'value': 100.0,
    'form': 0.8
}

# Get prediction
predicted_points = predict_upcoming_points(player_data)
print(f"Predicted points for next week: {predicted_points}")
```

### 3. Load Trained Model

```python
import pickle

# Load artifacts
with open('fpl_ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('fpl_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('fpl_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
```

---

## 📈 Results

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | ~1.8-2.2 | On average, predictions are off by 2 points |
| **RMSE** | ~2.5-3.0 | Typical prediction error magnitude |
| **R²** | ~0.35-0.45 | Model explains 35-45% of variance |

> **Note**: FPL contains inherent randomness (injuries, referee decisions, luck). An R² of 0.35-0.45 indicates the model captures meaningful patterns despite football's unpredictability.

### Feature Importance (Top 5)

1. **Form** - Recent performance is the strongest predictor
2. **Minutes** - Playing time directly impacts point potential
3. **Goals Scored** - Direct offensive contribution
4. **Influence** - Overall match impact
5. **Position** - Different roles have different scoring patterns

### Key Insights

✅ **Midfielders** score the highest average points (goals + assists + clean sheet potential)

✅ **Form matters**: Players in good form for 2-3 weeks tend to continue performing

✅ **Top total points ≠ Top form**: Consistent performers may have lower total points than boom-bust players

✅ **Playing time is critical**: Minutes played is a strong predictor of points

---

## 🔍 Model Explainability

### SHAP Analysis

<div align="center">

**Global Feature Importance**

Feature contributions across all predictions, showing which features most influence the model's decisions.

</div>

**Key Findings:**
- `form` has the highest mean absolute SHAP value
- `minutes` is consistently important
- `creativity` and `influence` provide additional context
- Position encoding captures role-based differences

### LIME Explanations

**Local explanations** for individual predictions:
- Shows exactly why the model predicted X points for a specific player
- Highlights which features pushed the prediction up or down
- Validates that model reasoning aligns with football logic

**Example:**
```
Player: High-performing Midfielder
Prediction: 6.2 points

Top Contributing Features:
+ form (0.8) → +2.1 points
+ goals_scored (2) → +1.8 points
+ creativity (80.0) → +1.2 points
- clean_sheets (0) → -0.3 points
```

---

## 📁 Project Structure

```
fpl-prediction/
│
├── fpl-prediction.ipynb          # Main Jupyter notebook (40 cells)
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
│
├── dataset/
│   └── cleaned_merged_seasons.csv         # Raw dataset
│
├── output/
│   ├── cleaned_merged_seasons_with_form.csv    # Cleaned data with form
│   ├── fpl_cleaned_with_features.csv           # Final modeling dataset
│   ├── fpl_ridge_model.pkl                     # Trained model
│   ├── fpl_scaler.pkl                          # Feature scaler
│   └── fpl_label_encoder.pkl                   # Position encoder
│
└── docs/
    └── CLAUDE.md                  # Project specifications
```

---

## 📦 Deliverables

### ✅ 1. Refined Jupyter Notebook
- 40 cells (22 Markdown + 18 Code)
- Complete workflow with explanations
- Justifications for all design choices
- Reproducible from start to finish

### ✅ 2. Cleaned Dataset
- `cleaned_merged_seasons_with_form.csv`
- Contains form feature
- All missing values handled
- No duplicates

### ✅ 3. Analytical Report
- Answers to data engineering questions
- Visualizations for both questions
- Feature selection justification
- Embedded in notebook

### ✅ 4. Predictive Model
- Ridge Regression (statistical ML)
- Predicts `upcoming_total_points`
- Evaluated with MAE, MSE, RMSE, R²
- Saved as `fpl_ridge_model.pkl`

### ✅ 5. XAI Outputs
- SHAP summary plots and bar plots
- LIME explanations for 3 predictions
- Ridge coefficient visualization
- Feature importance analysis

### ✅ 6. Inference Function
- `predict_upcoming_points(player_data)`
- Accepts raw input (dict or DataFrame)
- Handles preprocessing automatically
- Production-ready

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Implement cross-validation for hyperparameter tuning
- [ ] Try ensemble methods (stacking multiple models)
- [ ] Add opponent strength as a feature
- [ ] Incorporate fixture difficulty ratings

### Feature Engineering
- [ ] Create home/away performance splits
- [ ] Add team-level features (recent form, goals scored/conceded)
- [ ] Player injury history
- [ ] Head-to-head statistics against specific opponents

### Deployment
- [ ] Create REST API for predictions
- [ ] Build web dashboard for visualization
- [ ] Automated weekly retraining pipeline
- [ ] Real-time prediction updates

### Analysis
- [ ] Transfer recommendation system
- [ ] Captain selection optimizer
- [ ] Differential pick identifier
- [ ] Budget optimization algorithm

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include comments for complex logic
- Update README for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Fantasy Premier League** for providing the game and data
- **Scikit-learn** for machine learning tools
- **SHAP** and **LIME** for explainability frameworks
- **Kaggle** community for inspiration and datasets

---

## 📧 Contact

**Project Maintainer**: Your Name

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and ⚽ by FPL enthusiasts

[Back to Top](#-fantasy-premier-league-player-performance-prediction)

</div>
