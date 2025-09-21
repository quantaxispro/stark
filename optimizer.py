
import optuna
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, accuracy_score, log_loss, r2_score)
from sklearn.base import TransformerMixin

class LGBMRollingClassifier(TransformerMixin):
	"""LGBM 分類器"""

	def __init__(self) -> None:
		super(LGBMRollingClassifier, self).__init__()

	def objective(self, trial, X, y):
		# 參數網格
		params = {
			# 二分類目標函數與評估指標
			'objective': 'binary',
			'metric': 'binary_logloss',
			# 多分類配置（如需使用請取消註釋）
			# 'objective': 'multiclass',
			# 'metric': 'multi_logloss',
			# 'num_class': len(set(y)),  # 設定類別數量
			'class_weight': 'balanced',  # 處理不平衡資料
			'importance_type': 'gain',
			
			# 核心超參數
			"boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),  # 添加 dart 提高穩定性
			"num_leaves": trial.suggest_int("num_leaves", 20, 150, step=10),  # 擴大範圍
			"n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),  # 擴大範圍以找到更優解
			"learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),  # 使用對數尺度搜尋
			"max_depth": trial.suggest_int("max_depth", 3, 12, 1),  # 擴大深度範圍
			
			# 防止過擬合參數
			"min_child_samples": trial.suggest_int("min_child_samples", 10, 100, 5),  # 擴大範圍
			"subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),  # 調整下限，金融資料通常需要更多樣本
			"colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05),  # 特徵抽樣，減少過擬合
			"reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),  # L1 正則化，使用對數尺度
			"reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),  # L2 正則化，使用對數尺度
			
			# 金融時序資料特定參數
			"feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0, step=0.05),  # 每次迭代隨機選擇特徵比例
			"verbosity": -1  # 靜默模式
		}
		
		# 註：此參數空間複雜度（約 10 個超參數，多為連續值），建議 300-500 trials 較合適。1000 trials 可能過度搜尋，且收益遞減。若計算資源充足，可先設 300，觀察收斂再決定是否增加。

		# 滾動視窗訓練與預測
		tscv = TimeSeriesSplit(n_splits=5)

		cv_scores = np.empty(5)
		for idx, (train_index, test_index) in enumerate(tscv.split(X)):
			X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]
			y_train, y_val = y.iloc[train_index], y.iloc[test_index]

			# 建立 LightGBM 訓練集與驗證集
			train_data = lgb.Dataset(X_train, label=y_train)
			val_data = lgb.Dataset(X_val, label=y_val)

			# 訓練模型
			model = lgb.train(params, train_data, valid_sets=[train_data, val_data])

			# 預測驗證集
			# y_pred = np.argmax(model.predict(X_val), axis=1)
			# 二分類使用機率閾值 0.5
			y_prob = model.predict(X_val)
			y_pred = (y_prob >= 0.5).astype(int)

			# 計算準確率作為目標函數評估指標
			cv_scores[idx] = accuracy_score(y_val, y_pred)

		return np.mean(cv_scores)

	def optimizer(self, study_name: str = "lgb_rolling_clf", X=None, y=None):
		"""
		Optuna 超參數優化器（繁體中文註釋）
		
		參數說明：
		study_name：Optuna study 名稱，預設 'lgb_rolling_clf'
		X：訓練特徵（DataFrame，僅限數值型）
		y：訓練標籤
		
		流程：
		1. 建立 Optuna study（內存模式，避免資料庫檔案錯誤）
		2. 以 objective 方法進行超參數搜尋，預設 1000 trials
		3. 輸出最佳參數與分數
		4. 用最佳參數訓練 LightGBM 分類器，返回模型
		"""
		
		# 建立 Optuna study（內存模式）
		study = optuna.create_study(
			sampler=optuna.samplers.TPESampler(seed=42),
			study_name=study_name,
			direction="maximize",
			pruner=optuna.pruners.HyperbandPruner(),
			load_if_exists=True
		)

		# 執行超參數優化
		study.optimize(lambda trial: self.objective(trial, X, y), n_trials=1000)

		# 取得最佳超參數
		best_params = study.best_params
		print(f'best params: {study.best_params}')
		print(f'best value: {study.best_value}')

		# 用最佳參數訓練 LightGBM 模型
		model = lgb.LGBMClassifier(
			objective='binary',
			metric='binary_logloss',
			# num_class=len(set(y)),  # 設定類別數量
			class_weight='balanced',
			importance_type='gain',
			**best_params,
			random_state=42,
			verbosity=-1
		)

		model.fit(X, y)

		return model
	
