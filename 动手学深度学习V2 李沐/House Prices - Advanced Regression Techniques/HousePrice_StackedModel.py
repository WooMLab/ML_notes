# 导入一些最基本的库
import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cb

# ================= CONFIG =================
SEED = 42
FOLDS = 10 
DEBUG = False
N_ESTIMATORS = 2000 if not DEBUG else 200

LGB_BASE_PARAMS = {
    'n_estimators': N_ESTIMATORS,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': -1
}

# LGB 参数变体
# 其中测试过程中发现 dart变体的MSE值很高（即单模型表现不好），但是其加入后能够提高最后集成学习后的效果（个人猜测可以让模型学习到一些别的细节）
LGB_PARAM_VARIANTS = [
    {**LGB_BASE_PARAMS},
    {**LGB_BASE_PARAMS, 'num_leaves': 63, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 0.5},
    {**LGB_BASE_PARAMS, 'num_leaves': 15, 'learning_rate': 0.02, 'colsample_bytree': 0.3},
    {**LGB_BASE_PARAMS, 'num_leaves': 127, 'learning_rate': 0.005, 'n_estimators': 3000},
    {**LGB_BASE_PARAMS, 'objective': 'huber', 'alpha': 0.9},
    {**LGB_BASE_PARAMS, 'num_leaves': 47, 'colsample_bytree': 0.5, 'subsample': 0.7},
    {**LGB_BASE_PARAMS, 'boosting_type': 'dart', 'drop_rate': 0.1, 'skip_drop': 0.5}
]

LGB_SEEDS = [42, 7, 13]

CB_PARAM_VARIANTS = [
    {
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 200,
        'verbose': False
    },
    {
        'iterations': 2000,
        'learning_rate': 0.02,
        'depth': 8,
        'l2_leaf_reg': 1,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 7,
        'od_type': 'Iter',
        'od_wait': 200,
        'verbose': False
    },
    {
        'iterations': 3000,
        'learning_rate': 0.01,
        'depth': 5,
        'l2_leaf_reg': 5,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 13,
        'od_type': 'Iter',
        'od_wait': 300,
        'verbose': False
    },
    {
        'iterations': 2500,
        'learning_rate': 0.015,
        'depth': 7,
        'l2_leaf_reg': 2,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 21,
        'od_type': 'Iter',
        'od_wait': 250,
        'verbose': False
    },
    {
        'iterations': 2200,
        'learning_rate': 0.04,
        'depth': 6,
        'l2_leaf_reg': 2,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 31,
        'od_type': 'Iter',
        'od_wait': 300,
        'verbose': False,
        'task_type': 'CPU',
        'thread_count': -1
    }
]

TRAIN_PATH = './house-prices-advanced-regression-techniques/train.csv'
TEST_PATH  = './house-prices-advanced-regression-techniques/test.csv'

# 固定随机种子
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
seed_everything()

# ============= 数据加载 & 预处理  =============
if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    raise FileNotFoundError("请将train.csv/test.csv放入'./house-prices-advanced-regression-techniques/'目录")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train_ID = train['Id'].copy()
test_ID = test['Id'].copy()
y = np.log1p(train['SalePrice'].values)
train.drop(['SalePrice'], axis=1, inplace=True)

# ---------------- feature engineering ----------------
def feature_engineer(df):
    df = df.copy()
    df['TotalSF'] = df.get('TotalBsmtSF',0).fillna(0) + df.get('1stFlrSF',0).fillna(0) + df.get('2ndFlrSF',0).fillna(0)
    df['TotalPorchSF'] = (df.get('OpenPorchSF',0).fillna(0) + df.get('EnclosedPorch',0).fillna(0)
                         + df.get('3SsnPorch',0).fillna(0) + df.get('ScreenPorch',0).fillna(0)
                         + df.get('WoodDeckSF',0).fillna(0))
    df['TotalBath'] = df.get('BsmtFullBath',0).fillna(0) + df.get('FullBath',0).fillna(0) + 0.5*(df.get('BsmtHalfBath',0).fillna(0) + df.get('HalfBath',0).fillna(0))
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['SinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    df['HasPool'] = (df.get('PoolArea',0).fillna(0) > 0).astype(int)
    df['HasGarage'] = (df.get('GarageArea',0).fillna(0) > 0).astype(int)
    df['HasBasement'] = (df.get('TotalBsmtSF',0).fillna(0) > 0).astype(int)
    df['OverallQual_x_LivArea'] = df['OverallQual'] * df['GrLivArea']
    if 'LotArea' in df.columns:
        df['LotArea_log'] = np.log1p(df['LotArea'])
    df['OverallQual_x_TotalSF'] = df['OverallQual'] * df['TotalSF']
    return df

train = feature_engineer(train)
test = feature_engineer(test)

all_df = pd.concat([train, test], sort=False).reset_index(drop=True)

cat_cols = [c for c in all_df.columns if all_df[c].dtype == 'object']
for c in all_df.columns:
    if all_df[c].dtype in ['int64','int32'] and all_df[c].nunique() < 20 and c != 'Id':
        cat_cols.append(c)
exclude = set(cat_cols) | {'Id'}
num_cols = [c for c in all_df.columns if c not in exclude]

for c in cat_cols:
    all_df[c] = all_df[c].fillna('None').astype(str)
for c in num_cols:
    all_df[c] = all_df[c].fillna(all_df[c].median())

skews = all_df[num_cols].skew().abs().sort_values(ascending=False)
skewed_feats = skews[skews > 0.8].index.tolist()
print("Skewed numeric features (log1p applied):", skewed_feats)
for c in skewed_feats:
    all_df[c] = np.log1p(all_df[c].values)

lbl_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    all_df[c] = le.fit_transform(all_df[c].astype(str))
    lbl_encoders[c] = le

n_train = train.shape[0]
X_all = all_df.copy()
X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

def kfold_target_encode(train_series, target, test_series, n_splits=5, seed=SEED, min_samples_leaf=1, smoothing=10):
    oof = pd.Series(np.nan, index=train_series.index)
    test_encoded = np.zeros(len(test_series))
    prior = target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_series):
        tr_s = train_series.iloc[tr_idx]
        tr_y = target[tr_idx]
        stats = pd.DataFrame({'cat': tr_s, 'y': tr_y}).groupby('cat')['y'].agg(['mean','count'])
        smoothing_factor = 1 / (1 + np.exp(-(stats['count'] - min_samples_leaf) / smoothing))
        stats['te'] = prior * (1 - smoothing_factor) + stats['mean'] * smoothing_factor
        mapping = stats['te'].to_dict()
        oof.iloc[val_idx] = train_series.iloc[val_idx].map(mapping).fillna(prior)
        test_encoded += test_series.map(mapping).fillna(prior).values
    test_encoded /= n_splits
    return oof.values, test_encoded

high_card = [c for c in cat_cols if X_train[c].nunique() > 10]
high_card = sorted(high_card, key=lambda x: X_train[x].nunique(), reverse=True)[:6]
print("Target encoding features:", high_card)

for c in high_card:
    oof_te, test_te = kfold_target_encode(X_train[c], y, X_test[c], n_splits=FOLDS, seed=SEED, min_samples_leaf=20, smoothing=10)
    X_train[c + '_te'] = oof_te
    X_test[c + '_te'] = test_te
    num_cols.append(c + '_te')

features = [c for c in X_train.columns if c != 'Id']
print("Total features:", len(features))

# ============= 训练函数（CPU 版 LGB / CPU CatBoost） ==============
def lgb_train_oof(X, y, X_test, params, folds=FOLDS, seed=SEED):
    oof = np.zeros(len(X))
    test_preds_folds = np.zeros((len(X_test), folds))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = lgb.LGBMRegressor(**params)
        # CPU 运行，早停保留
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)]
            )
        best_iter = getattr(model, 'best_iteration_', None)
        if best_iter:
            oof[val_idx] = model.predict(X_val, num_iteration=best_iter)
            test_preds_folds[:, i] = model.predict(X_test, num_iteration=best_iter)
        else:
            oof[val_idx] = model.predict(X_val)
            test_preds_folds[:, i] = model.predict(X_test)
        print(f"LGB fold {i} rmse: {np.sqrt(mean_squared_error(y_val, oof[val_idx])):.6f}")
    return oof, test_preds_folds.mean(axis=1)

def cb_train_oof(X, y, X_test, params, cat_features_indices, folds=FOLDS, seed=SEED):
    oof = np.zeros(len(X))
    test_preds_folds = np.zeros((len(X_test), folds))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        train_pool = cb.Pool(X_tr, y_tr, cat_features=cat_features_indices)
        val_pool = cb.Pool(X_val, y_val, cat_features=cat_features_indices)
        model = cb.CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
        oof[val_idx] = model.predict(X_val)
        test_preds_folds[:, i] = model.predict(X_test)
        print(f"CatBoost fold {i} rmse: {np.sqrt(mean_squared_error(y_val, oof[val_idx])):.6f}")
    return oof, test_preds_folds.mean(axis=1)

# ============= 训练 LGB 模型（CPU）=============
lgb_oofs, lgb_tests = [], []
for p_idx, base_params in enumerate(LGB_PARAM_VARIANTS):
    for s in LGB_SEEDS:
        params = base_params.copy()
        params['random_state'] = s
        print(f"\n--- Training LGB_v{p_idx}_s{s} (CPU) ---")
        oof, testp = lgb_train_oof(X_train[features], y, X_test[features], params, folds=FOLDS, seed=s)
        lgb_oofs.append(oof)
        lgb_tests.append(testp)

# ============= 训练 CatBoost 模型（CPU）=============
cb_oofs, cb_tests = [], []
cat_features_indices = [i for i, col in enumerate(features) if col in cat_cols]
for v_idx, params in enumerate(CB_PARAM_VARIANTS):
    print(f"\n--- Training CB_v{v_idx} (CPU) ---")
    oof, testp = cb_train_oof(X_train[features], y, X_test[features], params, cat_features_indices, folds=FOLDS, seed=SEED)
    cb_oofs.append(oof)
    cb_tests.append(testp)

# ---------------- 堆叠融合（Ridge + 统计特征） ----------------
print("\n=== 构建Meta特征矩阵 ===")
# 合并LGB和CatBoost的OOF/测试预测
all_oofs = lgb_oofs + cb_oofs
all_tests = lgb_tests + cb_tests
meta_train = np.column_stack(all_oofs)
meta_test = np.column_stack(all_tests)
print(f"合并后Meta特征维度: {meta_train.shape}")

# 增加统计特征（每个样本在base模型上的统计量）
base_preds_train = np.column_stack(all_oofs)
base_preds_test = np.column_stack(all_tests)
stats_train = np.column_stack([
    base_preds_train.mean(axis=1),
    base_preds_train.std(axis=1),
    base_preds_train.min(axis=1),
    base_preds_train.max(axis=1),
    base_preds_train.max(axis=1) - base_preds_train.min(axis=1)
])
stats_test = np.column_stack([
    base_preds_test.mean(axis=1),
    base_preds_test.std(axis=1),
    base_preds_test.min(axis=1),
    base_preds_test.max(axis=1),
    base_preds_test.max(axis=1) - base_preds_test.min(axis=1)
])
meta_train_enhanced = np.column_stack([meta_train, stats_train])
meta_test_enhanced = np.column_stack([meta_test, stats_test])
print(f"添加统计特征后维度: {meta_train_enhanced.shape}")

# 方法1: Ridge回归（稳定优先）
print("\n=== 方法1: Ridge回归学习权重 ===")
meta_ridge = Ridge(alpha=1.0)
meta_ridge.fit(meta_train_enhanced, y)
ridge_oof = meta_ridge.predict(meta_train_enhanced)
ridge_test_pred_log = meta_ridge.predict(meta_test_enhanced)
ridge_cv_rmse = np.sqrt(mean_squared_error(y, ridge_oof))
print(f"Ridge Meta CV RMSE: {ridge_cv_rmse:.6f}")

# 方法2: NNLS（非负权重）
print("\n=== 方法2: NNLS学习权重 ===")
try:
    from scipy.optimize import nnls
    w, _ = nnls(meta_train, y)
    nnls_oof = meta_train.dot(w)
    nnls_test_pred_log = meta_test.dot(w)
    nnls_cv_rmse = np.sqrt(mean_squared_error(y, nnls_oof))
    print(f"NNLS Meta CV RMSE: {nnls_cv_rmse:.6f}")
    print(f"NNLS权重和: {w.sum():.4f}")
except ImportError:
    print("scipy未安装，跳过NNLS方法")
    nnls_cv_rmse = float('inf')
    nnls_test_pred_log = None

# 方法3: 简单平均（Baseline）
simple_avg_oof = meta_train.mean(axis=1)
simple_avg_test = meta_test.mean(axis=1)
simple_avg_cv_rmse = np.sqrt(mean_squared_error(y, simple_avg_oof))
print(f"\n=== 方法3: 简单平均 CV RMSE: {simple_avg_cv_rmse:.6f} ===")

# 选择最优方法
methods = {
    'Ridge': (ridge_cv_rmse, ridge_test_pred_log),
    'NNLS': (nnls_cv_rmse, nnls_test_pred_log) if nnls_test_pred_log is not None else (float('inf'), None),
    'SimpleAvg': (simple_avg_cv_rmse, simple_avg_test)
}
best_method = min(methods.keys(), key=lambda x: methods[x][0])
best_rmse, final_test_pred_log = methods[best_method]
print(f"\n最佳方法: {best_method}, CV RMSE: {best_rmse:.6f}")
if best_method == 'Ridge':
    print("使用Ridge回归（含统计特征）作为最终预测")
elif best_method == 'NNLS':
    print("使用NNLS（仅原始预测）作为最终预测")
else:
    print("使用简单平均作为最终预测")

# ---------------- 自适应取整策略 ----------------
def adaptive_rounding(prices, oof_prices=None, oof_true=None):
    if oof_prices is not None and oof_true is not None:
        best_rmse = float('inf')
        best_thresholds = None
        t1_candidates = [80000, 90000, 100000, 110000, 120000]
        t2_candidates = [180000, 190000, 200000, 210000, 220000]
        for t1 in t1_candidates:
            for t2 in t2_candidates:
                rounded = oof_prices.copy()
                mask1 = oof_prices < t1
                mask2 = (oof_prices >= t1) & (oof_prices < t2)
                mask3 = oof_prices >= t2
                rounded[mask1] = np.round(rounded[mask1] / 10) * 10
                rounded[mask2] = np.round(rounded[mask2] / 50) * 50
                rounded[mask3] = np.round(rounded[mask3] / 100) * 100
                rmse = np.sqrt(mean_squared_error(np.log1p(oof_true), np.log1p(rounded)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_thresholds = (t1, t2)
        print(f"最优取整阈值: <{best_thresholds[0]}: 10, {best_thresholds[0]}-{best_thresholds[1]}: 50, >{best_thresholds[1]}: 100")
        print(f"取整后OOF RMSE: {best_rmse:.6f}")
        t1, t2 = best_thresholds
        result = prices.copy()
        mask1 = prices < t1
        mask2 = (prices >= t1) & (prices < t2)
        mask3 = prices >= t2
        result[mask1] = np.round(result[mask1] / 10) * 10
        result[mask2] = np.round(result[mask2] / 50) * 50
        result[mask3] = np.round(result[mask3] / 100) * 100
        return result.astype(int)
    else:
        result = prices.copy()
        mask1 = prices < 100000
        mask2 = (prices >= 100000) & (prices < 200000)
        mask3 = prices >= 200000
        result[mask1] = np.round(result[mask1] / 10) * 10
        result[mask2] = np.round(result[mask2] / 50) * 50
        result[mask3] = np.round(result[mask3] / 100) * 100
        return result.astype(int)

# 逆变换+取整
final_test_pred_float = np.expm1(final_test_pred_log)
if 'ridge_oof' in locals():
    oof_prices = np.expm1(ridge_oof)
    oof_true = np.expm1(y)
    final_test_pred = adaptive_rounding(final_test_pred_float, oof_prices, oof_true)
else:
    final_test_pred = (np.round(final_test_pred_float / 10) * 10).astype(int)
print("\nPost-processing: 自适应取整已应用")

# 生成提交文件
sub = pd.DataFrame({'Id': test_ID, 'SalePrice': final_test_pred})
sub.to_csv('submission_opt.csv', index=False)
print("已保存 submission_opt.csv")
print("\n预测样本示例:")
print(sub.head())