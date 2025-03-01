import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


train_data = pd.read_csv('stage_1_regression_train.csv')
test_data = pd.read_csv('stage_1_regression_test_features.csv')

X_train = train_data.drop('target', axis=1)
y_train = np.log1p(train_data['target'])
X_test = test_data.drop('index', axis=1)

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 2. Оптимизация гиперпараметров для CatBoost ===
def catboost_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1500, 5000),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-2, 10),
        'random_strength': trial.suggest_uniform('random_strength', 0, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'grow_policy': trial.suggest_categorical('grow_policy', ['Depthwise', 'SymmetricTree']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian']),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.5, 1.0),
        'random_state': 42
    }

    # Удаляем строки с NaN в целевой переменной
    valid_indices = ~np.isnan(y_train)
    X_train_clean = X_train[valid_indices]
    y_train_clean = y_train[valid_indices]

    # Разделение на обучение и валидацию
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_clean, y_train_clean, test_size=0.2, random_state=42)

    # Обучение CatBoost
    model = CatBoostRegressor(**params, verbose=0)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    preds = model.predict(X_val)

    # Вычисление RMSE
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse


# Оптимизация гиперпараметров CatBoost
catboost_study = optuna.create_study(direction='minimize')
catboost_study.optimize(catboost_objective, n_trials=100)
print("Лучшие параметры CatBoost:", catboost_study.best_params)

# === 3. Оптимизация гиперпараметров для LightGBM ===
def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# Оптимизация гиперпараметров LightGBM
lgbm_study = optuna.create_study(direction='minimize')
lgbm_study.optimize(lgbm_objective, n_trials=100)
print("Лучшие параметры LightGBM:", lgbm_study.best_params)

# === 4. Обучение Stacking ансамбля ===
# Базовые модели
catboost = CatBoostRegressor(**catboost_study.best_params, verbose=0, random_state=42)
lgbm = LGBMRegressor(**lgbm_study.best_params, random_state=42)

# Верхнеуровневая модель (Ridge)
final_estimator = Ridge(alpha=1.0, random_state=42)

# Кросс-валидация
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Stacking ансамбль
stacking_model = StackingRegressor(
    estimators=[('catboost', catboost), ('lgbm', lgbm)],
    final_estimator=final_estimator,
    cv=kf
)

# Обучение Stacking модели
stacking_model.fit(X_train, y_train)

# === 5. Предсказание и сохранение результатов ===
# Предсказания
final_predictions = np.expm1(stacking_model.predict(X_test))  # Обратное логарифмирование

# Сохранение результатов
output = pd.DataFrame({'index': test_data['index'], 'target': final_predictions})
output.to_csv('stage_1_regression_test_target.csv', index=False)

print("Файл с предсказаниями сохранён: stage_1_regression_test_target.csv")
