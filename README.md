
# Fantasy Premier League Prediction Model

## Project Overview
Machine learning pipeline to predict Fantasy Premier League player points for upcoming gameweeks using historical performance data.

## Dataset
- **Source:** [Kaggle - Fantasy Football](https://www.kaggle.com/datasets/jaskirat/fantasy-football)
- **Records:** 96,169 player-gameweek observations  
- **Seasons:** 2016-17, 2017-18, 2020-21, 2021-22, 2022-23

## 🔗 Live Notebook
**Kaggle:** [Link to your Kaggle notebook - make it public after deadline]

## Project Structure
```

├── models/          # Trained ML models
├── outputs/         # Cleaned data and results
├── notebooks/       # Jupyter notebook
├── README.md
├── requirements.txt
└── ANALYTICAL_REPORT.md

````

## Installation
```bash
git clone https://github.com/hussienhaithamm/fpl-prediction-project.git
cd fpl-prediction-project
pip install -r requirements.txt
````

---

## 🧮 Results

### Model Performance (2022-23 Season)

| Model             | MAE   | RMSE  | R²    | Training Time |
| ----------------- | ----- | ----- | ----- | ------------- |
| Gradient Boosting | 1.193 | 2.139 | 0.293 | 21.19s        |
| FFNN              | 1.176 | 2.144 | 0.289 | 204.56s       |
| Linear Regression | 1.223 | 2.149 | 0.286 | 0.13s         |
| Random Forest     | 1.200 | 2.153 | 0.283 | 12.24s        |

**Best Model:** Gradient Boosting (RMSE: 2.139, R²: 0.293)

---

## Key Findings

**Question A: Which positions score most points?**

| Position | Avg Points/Season |
| -------- | ----------------- |
| MID      | 11,384            |
| DEF      | 8,705             |
| FWD      | 3,976             |
| GK       | 2,449             |

**Question B: Top 5 Players (2022-23)**

1. Erling Haaland (FWD): 272 pts
2. Harry Kane (FWD): 263 pts
3. Mohamed Salah (MID): 239 pts
4. Martin Ødegaard (MID): 212 pts
5. Marcus Rashford (MID): 205 pts

**Consistency Finding:** 4 out of 5 top scorers also had top 5 form ratings.

---

## Feature Importance

Top predictive features:

1. `minutes` (45.7%)
2. `selected` (14.4%)
3. `value` (8.2%)
4. `form` (7.7%)
5. `creativity` (4.6%)

---

## Deliverables

✅ 1. Jupyter Notebook - Complete with all analysis
✅ 2. Cleaned Dataset - `outputs/cleaned_dataset_with_form.csv`
✅ 3. Analytical Report - `ANALYTICAL_REPORT.md`
✅ 4. Trained Models - All 4 models in `models/`
✅ 5. XAI Outputs - SHAP & LIME visualizations in notebook
✅ 6. Inference Function - Production-ready prediction function

---

## Usage Example

```python
# Load models and make predictions
from predict import predict_upcoming_points

player = {
    'assists': 2, 'bonus': 3, 'bps': 45, 'clean_sheets': 1,
    'goals_conceded': 0, 'goals_scored': 1, 'minutes': 90,
    'own_goals': 0, 'penalties_missed': 0, 'penalties_saved': 0,
    'red_cards': 0, 'saves': 0, 'yellow_cards': 0, 'total_points': 12,
    'position': 'MID', 'creativity': 65.5, 'influence': 75.2,
    'threat': 58.0, 'ict_index': 19.8, 'value': 95,
    'selected': 450000, 'form': 1.05
}

predicted_points = predict_upcoming_points(player)
print(f"Predicted points: {predicted_points:.2f}")
```

---

## Technologies Used

* Python 3.11
* pandas, numpy
* scikit-learn
* TensorFlow / Keras
* SHAP, LIME
* matplotlib, seaborn

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

````
