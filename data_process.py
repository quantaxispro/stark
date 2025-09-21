#  數據預處理函數
import polars as pl
import json


def data_preprocessing(json_path='data.json'):
    """數據預處理函數"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # --------------------------------------------------------
    # 需要合併的季度財務數據欄位
    q_data_cols = ['financialGrowth', 'ratios', 'cashFlowStatementGrowth',
                   'incomeStatementGrowth', 'balanceSheetStatementGrowth']
    
    # 將所有季度財務數據轉為 DataFrame
    q_dfs = [pl.DataFrame(data[col]) for col in q_data_cols]
    
    # 依主鍵合併所有季度財務數據
    q_df = q_dfs[0]
    for i, df in enumerate(q_dfs[1:]):
        q_df = q_df.join(
            df,
            on=['symbol', 'date', 'calendarYear', 'period'],
            how='inner'
        )
    # --------------------------------------------------------
    
    # 需要處理的日線技術指標欄位
    d_data_cols = ['tech5', 'tech20', 'tech60', 'tech252']
    # 日線合併主鍵
    d_meger_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # 將所有日線技術指標數據轉為 DataFrame
    d_dfs = [pl.DataFrame(data[col]) for col in d_data_cols]
    d_df = d_dfs[0]
    for i, df in enumerate(d_dfs[1:]):
        d_df = d_df.join(
            df,
            on=d_meger_cols,
            how='inner',
            suffix=f'_{i+1}'
        )
    
    # 日期欄位只保留前10位（yyyy-mm-dd），並轉為日期型別
    d_df = d_df.with_columns([
        pl.col('date').str.slice(offset=0, length=10)
    ]).with_columns(pl.col('date').str.strptime(pl.Date, "%Y-%m-%d"))
    
    # 讀取歷史行情數據並轉換日期型別
    his_df = pl.DataFrame(data['historicalPriceFull']['historical']).with_columns(pl.col('date').str.strptime(pl.Date, "%Y-%m-%d")) # 日期轉換
    # 合併日線行情與技術指標，並補上 symbol 欄位
    daily_df = his_df.join(d_df, on=d_meger_cols, how='inner').with_columns(pl.lit('1101.TW').alias('symbol'))  # daily_df 构建
    
    # --------------------------------------------------------
    # 季度數據處理，生成下一季度報告日期
    quarterly_df = q_df.sort(["symbol", "date"]).with_columns(pl.col('date').str.strptime(pl.Date, "%Y-%m-%d")).with_columns([
        pl.col("date").shift(-1).over("symbol").alias("next_report_date") # 下一季度報告日期
    ]).rename({'date':'report_date'})   # 報告日期
    
    # --------------------------------------------------------
    # 合併日線與季度數據，僅保留每個財報區間內的日線數據
    meger_df = daily_df.join(quarterly_df, on="symbol", how="left").filter(
        (pl.col("date") >= pl.col("report_date")) & 
        ((pl.col("next_report_date").is_null()) | (pl.col("date") < pl.col("next_report_date")))
    ).drop(['report_date', 'next_report_date','label','calendarYear','open','high','low','volume']) # 合併後過濾日期
    
    return meger_df.sort(['symbol', 'date'])