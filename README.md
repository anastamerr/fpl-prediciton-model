# Fantasy Premier League Player Performance Prediction

## Project Overview
Machine learning pipeline to predict Fantasy Premier League (FPL) player points for upcoming gameweeks using historical performance data and Ridge Regression.

## Dataset
- **Source:** [Kaggle - Fantasy Football](https://www.kaggle.com/datasets/jaskirat/fantasy-football)
- **Records:** 96,169 player-gameweek observations  
- **Players:** 1,327 unique players
- **Seasons:** 2016-17, 2017-18, 2020-21, 2021-22, 2022-23

## 🔗 Live Notebook
**Kaggle:** (https://www.kaggle.com/code/anastamerr/fpl-prediction-model0206af2223)

## Project Structure
```
fpl-prediction/
│
├── fpl-prediction.ipynb          # Main Jupyter notebook (40 cells)
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── dataset/
│   └── cleaned_merged_seasons.csv
│
├── output/
│   ├── cleaned_merged_seasons_with_form.csv
│   ├── fpl_cleaned_with_features.csv
│   ├── fpl_ridge_model.pkl
│   ├── fpl_scaler.pkl
│   └── fpl_label_encoder.pkl
```

## Installation
```bash
git clone https://github.com/anastamerr/fpl-prediciton-model.git
cd fpl-prediction
pip install -r requirements.txt
```

---

## 🧮 Results

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 1.2842 | On average, predictions are off by 1.28 points |
| **RMSE** | 2.2164 | Typical prediction error magnitude |
| **R²** | 0.2750 | Model explains 27.5% of variance |

**Model:** Ridge Regression (L2 regularized linear regression, α = 1.0)

> **Note:** FPL contains inherent randomness (injuries, referee decisions, luck). An R² of 0.275 indicates the model captures meaningful patterns despite football's unpredictability.

---

## Key Findings

### Question 1: Which positions score most points on average?

| Position | Avg Points per Gameweek |
|----------|-------------------------|
| FWD      | 1.62                    |
| MID      | 1.50                    |
| DEF      | 1.33                    |
| GK       | 1.21                    |

**Insight:** Forwards score the highest average points per gameweek (1.62), followed closely by Midfielders (1.50). However, Midfielders have more opportunities to score due to higher selection in teams.

### Question 2: Top Players Performance Evolution (2022-23)

**Top 5 Players by Total Points:**
1. Erling Haaland (272 pts)
2. Harry Kane (263 pts)
3. Mohamed Salah (239 pts)
4. Martin Ødegaard (212 pts)
5. Marcus Rashford (205 pts)

**Top 5 Players by Average Form:**
1. Erling Haaland (0.751)
2. Harry Kane (0.661)
3. Mohamed Salah (0.642)
4. Martin Ødegaard (0.562)
5. Gabriel Martinelli Silva (0.552)

**Key Finding:** 4 out of 5 top total scorers also had top 5 form ratings (Haaland, Kane, Salah, Ødegaard). This shows consistency is highly correlated with total success.

---

## Feature Importance

Top predictive features based on Ridge Regression coefficients:

| Rank | Feature | Coefficient | Impact |
|------|---------|-------------|--------|
| 1 | **minutes** | 0.769 | Highest - Playing time is critical |
| 2 | **form** | 0.408 | Strong - Recent performance matters |
| 3 | **value** | 0.395 | Strong - Player price reflects quality |
| 4 | **influence** | 0.070 | Moderate - Overall match impact |
| 5 | **creativity** | 0.065 | Moderate - Chance creation ability |

**Note:** Negative coefficients for goals_scored, assists, and clean_sheets are due to multicollinearity and interaction with other features. The model still uses these features effectively in combination.

---

## Key Insights

✅ **Forwards** score the highest average points per gameweek (1.62), though Midfielders have more total opportunities

✅ **Playing time is king**: Minutes played has the strongest coefficient (0.769) in the model

✅ **Form feature works**: Recent 4-game average (form) is the 2nd strongest predictor (0.408 coefficient)

✅ **Consistency matters**: 4 out of 5 top scorers in 2022-23 also had top 5 form ratings

✅ **Model captures patterns**: R² of 0.275 shows the model identifies meaningful trends despite football's randomness

---

## Deliverables

✅ 1. **Refined Jupyter Notebook** - 40 cells with complete workflow
✅ 2. **Cleaned Dataset** - `cleaned_merged_seasons_with_form.csv`
✅ 3. **Analytical Report** - Answers to data engineering questions
✅ 4. **Predictive Model** - Ridge Regression saved as `fpl_ridge_model.pkl`
✅ 5. **XAI Outputs** - SHAP & LIME visualizations
✅ 6. **Inference Function** - Production-ready `predict_upcoming_points()`

---

## Usage Example

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
print(f"Predicted points for next week: {predicted_points:.2f}")
```

---

## Technologies Used

* Python 3.8+
* pandas, numpy
* scikit-learn (Ridge Regression)
* SHAP, LIME (Explainability)
* matplotlib, seaborn

---

## Methodology

### 1. Data Cleaning
- Remove unnecessary columns (transfers, selection %, fixture IDs)
- Handle missing values (fill numeric with 0)
- Remove duplicates and sort chronologically

### 2. Feature Engineering
**Form Feature:**
```
form = (average total_points over past 4 gameweeks) / 10
```

### 3. Predictive Modeling
**Target Variable:**
```python
upcoming_total_points = shift(total_points, -1 week)
```

**Features Used:**
- Match-related: goals_scored, assists, minutes, clean_sheets
- Player-related: position, creativity, influence, value
- Engineered: form

### 4. Model Explainability
- **SHAP** - Global feature importance
- **LIME** - Local explanations for individual predictions
- Ridge coefficient visualization

---

## Course Information

**Course:** CSEN 903 - Systems and Machine Learning  
**Institution:** German University in Cairo  
**Instructor:** Dr. Nourhan Ehab  
**Semester:** Winter 2025

---

## Authors

* **Anas Tamer Saeed Osman** (9MET P017 55-11997) - [anas.osman@student.guc.edu.eg](mailto:anas.osman@student.guc.edu.eg)
* **Hussien Haitham Hussien Abdelmotaleb Hussien** (9MET P017 55-6592) - [hussien.hussien@student.guc.edu.eg](mailto:hussien.hussien@student.guc.edu.eg)
* **Ahmed Hany Mohamed Reda Abdulhamid** (9MET P009 55-8524) - [ahmed.abdulhamid@student.guc.edu.eg](mailto:ahmed.abdulhamid@student.guc.edu.eg)
* **Omar Khaled Mohamed Elhady Hassan Abdelaal** (9DMET P029 46-14114) - [omar.alhadi@student.guc.edu.eg](mailto:omar.alhadi@student.guc.edu.eg)

---

## License

Educational project for academic purposes.