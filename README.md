# # 🏡 House Price Prediction using Stacked Models and Feature Engineering

This project presents an end-to-end machine learning pipeline to predict house prices using the popular [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The approach leverages **stacking ensemble models** and advanced **feature engineering**, including **polynomial feature generation**.

---

## 📌 Project Highlights

- ✅ Full ML pipeline from EDA to model deployment
- 🔧 Robust data cleaning and preprocessing
- 🧠 Advanced feature engineering including polynomial features and interaction terms
- 🤖 Stacking ensemble of multiple base models (e.g., RidgeCV, RandomForest, LightGBM, XGBoost, CatBoost, etc.)
- 🧪 Hyperparameter tuning using `GridSearchCV`
- 💾 Model serialized using `joblib` for real-world use

---

## 🧰 Tech Stack

- **Languages**: Python 3.9+
- **Libraries**: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, seaborn, matplotlib, joblib
- **Modeling**: StackingRegressor, RidgeCV, RandomForest, BaggingRegressor, LightGBM, XGBoost, CatBoost
- **Feature Engineering**: Polynomial features, interaction terms, one-hot encoding, standardization
- **Tuning**: GridSearchCV

