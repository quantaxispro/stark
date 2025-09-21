from sklearn.base import TransformerMixin
import lightgbm as lgb
import pandas as pd
import numpy as np

class LGBMFeatureSelector(TransformerMixin):
    """基于LightGBM的特征选择器
    
    Args:
        model_type (str): 模型类型，'reg' 用于回归，'clf' 用于分类
        itype (str): 特征重要性类型，可选 'gain', 'split'
        n (int): 要选择的特征数量
        random_state (int, optional): 随机种子. Defaults to 42.
    """
    def __init__(self, model_type: str = 'reg', itype: str = 'gain', 
                 n: int = 5, random_state: int = 42) -> None:
        super().__init__()
        if model_type not in ['reg', 'clf']:
            raise ValueError("model_type must be either 'reg' or 'clf'")
        if itype not in ['gain', 'split']:
            raise ValueError("itype must be either 'gain' or 'split'")
        if n < 1:
            raise ValueError("n must be positive")
            
        self.model_type = model_type
        self.itype = itype
        self.n = n
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """拟合方法（为保持接口一致）"""
        return self

    def transform(self, X, y):
        """执行特征选择转换
        
        Args:
            X (pd.DataFrame): 输入特征矩阵
            y: 目标变量
            
        Returns:
            pd.DataFrame: 选择后的特征矩阵
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # 配置模型参数
        params = {
            'importance_type': self.itype,
            'n_estimators': 1000,
            'random_state': self.random_state,
            'verbosity': -1
        }
        
        # 根据模型类型选择合适的模型
        if self.model_type == 'reg':
            model = lgb.LGBMRegressor(**params)
        else:  # clf
            model = lgb.LGBMClassifier(class_weight='balanced', **params)

        # 训练模型并处理可能的异常
        try:
            model.fit(X, y)
        except Exception as e:
            raise ValueError(f"Error fitting LightGBM model: {str(e)}")

        # 计算并存储特征重要性
        self.feature_importances_ = dict(zip(model.feature_name_, model.feature_importances_))
        self.selected_features_ = [k for k, v in sorted(
            self.feature_importances_.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.n]]

        return X.loc[:, self.selected_features_]