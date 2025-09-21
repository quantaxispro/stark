
import polars as pl
import numpy as np

def backtesting(df: pl.DataFrame, rf: float = 0.0):
	"""
	簡單日線分類信號回測函數
	參數:
		df: 必須包含 ["date", "symbol", "y_pred", "return"]
		rf: 無風險利率 (預設 0)
		
	返回:
		result: dict (策略收益指標)
		df_out: 含策略收益和淨值的DataFrame
	"""
	# 按symbol、date排序，避免亂序
	df = df.sort(["symbol", "date"])

	# 將預測滯後一天，對齊到未來收益
	df = df.with_columns(
		pl.col("return").shift(-1).over("symbol").alias("next_return")
	)

	# 策略收益 = 前一日預測 * 當日真實收益
	df = df.with_columns(
		(pl.col("y_pred") * pl.col("next_return")).alias("strategy_return")
	)

	# 計算淨值曲線
	df = df.with_columns(
		(pl.col("strategy_return") + 1).cum_prod().alias("equity_curve")
	)

	# 轉 numpy 計算指標
	strategy_returns = df["strategy_return"].fill_null(0).to_numpy()

	# 年化收益率 (假設 252 個交易日)
	mean_ret = np.nanmean(strategy_returns)
	ann_return = mean_ret * 252

	# 年化波動率
	vol = np.nanstd(strategy_returns)
	ann_vol = vol * np.sqrt(252)

	# 夏普比率
	sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

	# 最大回撤
	equity = (df["strategy_return"].fill_null(0) + 1).cum_prod()
	peak = equity.cum_max()
	dd = (equity - peak) / peak
	max_drawdown = dd.min()

	result = {
		"Annual Return": ann_return,
		"Annual Volatility": ann_vol,
		"Sharpe Ratio": sharpe,
		"Max Drawdown": max_drawdown
	}

	return result, df


