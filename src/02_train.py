"""
02_train.py
01_preprocess.py가 만든 피처 CSV로 분류 모델을 학습하고, 가장 좋은 모델을 저장합니다.

- quick : RandomForest만 GridSearch
- full  : RandomForest / ExtraTrees / XGBoost / LightGBM / Stacking 비교 (CV macro-F1 기준)

결과물 (models/ 폴더):
- rf_model.pkl           : 최고 모델 + LabelEncoder + 피처 목록
- rf_model_result.png    : Confusion Matrix + Feature Importance
- rf_model_samples_*.png : 테스트셋 성공/실패 사례 이미지

사용법:
    python src/02_train.py --mode full
"""
import os
import time
import pickle
import platform
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = {
    'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic',
}.get(platform.system(), 'NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 인자 ──
parser = argparse.ArgumentParser(description='스쿼트 자세 분류 모델 학습')
parser.add_argument('--csv',      default=os.path.join(PROJECT_ROOT, 'data', 'features_orig.csv'))
parser.add_argument('--out',      default=os.path.join(PROJECT_ROOT, 'models', 'rf_model.pkl'))
parser.add_argument('--mode',     default='full', choices=['quick', 'full'])
parser.add_argument('--data_dir', default=os.path.join(PROJECT_ROOT, 'data', 'images'),
                    help='원본 이미지 폴더 (성공/실패 사례 시각화용)')
args, _ = parser.parse_known_args()

CSV_PATH   = args.csv
MODEL_PATH = args.out
MODE       = args.mode
DATA_DIR   = args.data_dir


# ============================================
# 데이터 로드
# ============================================
df = pd.read_csv(CSV_PATH)
print(f'CSV: {CSV_PATH}')
print(f'전체 샘플: {len(df)}장')
print(f'\n[클래스 분포]\n{df["label"].value_counts()}\n')

FEATURE_COLS = [
    'knee_L', 'knee_R', 'hip_L', 'hip_R',
    'ankle_L', 'ankle_R', 'cave_L', 'cave_R',
    'heel_lift_L', 'heel_lift_R',
    'torso_L', 'torso_R', 'sym_knee', 'knee_depth',
]

X     = df[FEATURE_COLS].values
y_str = df['label'].values

le = LabelEncoder()
y  = le.fit_transform(y_str)
print(f'클래스 순서: {list(le.classes_)}\n')

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y
)
df_test = df.loc[idx_test].reset_index(drop=True)
print(f'학습: {len(X_train)} / 테스트: {len(X_test)}\n')

sw_train = compute_sample_weight('balanced', y_train)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================
# 모델별 GridSearch
# ============================================
def tune_rf():
    grid = {
        'n_estimators':      [300, 500],
        'max_depth':         [None, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf':  [1, 2],
    }
    base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    return GridSearchCV(base, grid, cv=skf, scoring='f1_macro', n_jobs=-1, verbose=1)


def tune_et():
    grid = {
        'n_estimators':      [300, 500],
        'max_depth':         [None, 20, 30],
        'min_samples_split': [2, 5],
    }
    base = ExtraTreesClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    return GridSearchCV(base, grid, cv=skf, scoring='f1_macro', n_jobs=-1, verbose=1)


def tune_xgb():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print('xgboost 미설치 — 건너뜀')
        return None
    grid = {
        'n_estimators':     [300, 500],
        'max_depth':        [4, 6, 8],
        'learning_rate':    [0.05, 0.1],
        'subsample':        [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    base = XGBClassifier(
        objective='multi:softprob', eval_metric='mlogloss',
        random_state=42, n_jobs=-1, tree_method='hist',
    )
    return GridSearchCV(base, grid, cv=skf, scoring='f1_macro', n_jobs=-1, verbose=1)


def tune_lgbm():
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print('lightgbm 미설치 — 건너뜀')
        return None
    grid = {
        'n_estimators':  [300, 500],
        'num_leaves':    [31, 63],
        'max_depth':     [-1, 10, 20],
        'learning_rate': [0.05, 0.1],
        'subsample':     [0.8, 1.0],
    }
    base = LGBMClassifier(
        class_weight='balanced', random_state=42, n_jobs=-1, verbosity=-1,
    )
    return GridSearchCV(base, grid, cv=skf, scoring='f1_macro', n_jobs=-1, verbose=1)


def build_stacking(estimators_dict):
    estimators = [(name, est) for name, est in estimators_dict.items() if est is not None]
    if len(estimators) < 2:
        return None
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced'),
        cv=skf, n_jobs=-1, passthrough=False,
    )


# ============================================
# 실행
# ============================================
results = {}


def fit_and_score(name, search_or_model, fit_kwargs=None):
    print('=' * 60)
    print(f'{name} 튜닝/학습 중...')
    print('=' * 60)
    fit_kwargs = fit_kwargs or {}
    t0 = time.time()
    search_or_model.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.time() - t0

    if isinstance(search_or_model, GridSearchCV):
        model  = search_or_model.best_estimator_
        cv_f1  = search_or_model.best_score_
        print(f'  best params: {search_or_model.best_params_}')
    else:
        model = search_or_model
        cv_f1 = cross_val_score(model, X_train, y_train,
                                cv=skf, scoring='f1_macro', n_jobs=-1).mean()

    y_pred   = model.predict(X_test)
    test_acc = (y_pred == y_test).mean()
    test_f1  = f1_score(y_test, y_pred, average='macro')

    print(f'  CV F1(macro)   : {cv_f1:.4f}')
    print(f'  Test acc       : {test_acc*100:.2f}%')
    print(f'  Test F1(macro) : {test_f1:.4f}')
    print(f'  소요 시간      : {elapsed/60:.1f}분\n')

    results[name] = (model, cv_f1, test_acc, test_f1)
    return model


# 1) RandomForest
rf_search = tune_rf()
rf_best   = fit_and_score('RandomForest', rf_search)

et_best = xgb_best = lgbm_best = None
if MODE == 'full':
    et_search  = tune_et()
    et_best    = fit_and_score('ExtraTrees', et_search)

    xgb_search = tune_xgb()
    if xgb_search is not None:
        xgb_best = fit_and_score('XGBoost', xgb_search,
                                 fit_kwargs={'sample_weight': sw_train})

    lgbm_search = tune_lgbm()
    if lgbm_search is not None:
        lgbm_best = fit_and_score('LightGBM', lgbm_search)

    stacking = build_stacking({
        'rf': rf_best, 'et': et_best,
        'xgb': xgb_best, 'lgbm': lgbm_best,
    })
    if stacking is not None:
        fit_and_score('Stacking', stacking)


# ============================================
# 최고 모델 선택
# ============================================
best_name  = max(results, key=lambda n: results[n][1])
best_model, best_cv_f1, best_test_acc, best_test_f1 = results[best_name]

print('=' * 60)
print('모델 비교 결과 (CV F1 기준)')
print('=' * 60)
print(f'{"모델":<14} {"CV_F1":>8} {"Test_Acc":>10} {"Test_F1":>10}')
print('-' * 50)
for name in sorted(results, key=lambda n: -results[n][1]):
    _, cf, ta, tf = results[name]
    star = '  ★' if name == best_name else ''
    print(f'{name:<14} {cf:>8.4f} {ta*100:>9.2f}% {tf:>10.4f}{star}')
print()

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))


# ============================================
# 저장
# ============================================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
checkpoint = {
    'model':         best_model,
    'label_encoder': le,
    'features':      FEATURE_COLS,
    'meta': {
        'best_model_name': best_name,
        'cv_f1_macro':     best_cv_f1,
        'test_acc':        best_test_acc,
        'test_f1_macro':   best_test_f1,
        'csv_used':        CSV_PATH,
    },
}
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(checkpoint, f)
print(f'모델 저장: {MODEL_PATH}\n')


# ============================================
# 시각화 1: Confusion Matrix + Feature Importance
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap='Blues', ax=axes[0])
axes[0].set_title(f'{best_name} — Confusion Matrix\n(acc={best_test_acc*100:.2f}%)',
                  fontsize=13)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

if hasattr(best_model, 'feature_importances_'):
    importance_vals = best_model.feature_importances_
    title = f'{best_name} — Feature Importance'
elif best_name == 'Stacking' and hasattr(rf_best, 'feature_importances_'):
    importance_vals = rf_best.feature_importances_
    title = 'RandomForest (base) — Feature Importance'
else:
    importance_vals = None

if importance_vals is not None:
    imp = (pd.DataFrame({'feature': FEATURE_COLS, 'importance': importance_vals})
             .sort_values('importance'))
    axes[1].barh(imp['feature'], imp['importance'], color='steelblue')
    axes[1].set_title(title, fontsize=13)
    axes[1].set_xlabel('Importance')
    print('[피처 중요도 Top 5]')
    print(imp.tail(5).to_string(index=False))
else:
    axes[1].axis('off')

plt.tight_layout()
out_png = MODEL_PATH.replace('.pkl', '_result.png')
plt.savefig(out_png, dpi=120)
print(f'\n시각화 저장: {out_png}')
plt.show()


# ============================================
# 시각화 2: 성공 / 실패 이미지 사례
# ============================================
def show_sample_images(df_test, y_test, y_pred, le, data_dir,
                        n_per_class=4, save_path=None):
    """
    성공 / 실패 이미지를 각각 별도 figure로 깔끔하게 출력
    """
    classes        = list(le.classes_)
    correct_mask   = (y_pred == y_test)
    incorrect_mask = ~correct_mask

    def load_img(row):
        img_path = os.path.join(data_dir, row['label'], row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def draw_grid(title, mask, border_color, emoji):
        samples = []
        for cls in classes:
            cls_idx = np.where((y_test == le.transform([cls])[0]) & mask)[0]
            for i in cls_idx[:n_per_class]:
                row        = df_test.iloc[i]
                pred_label = le.classes_[y_pred[i]]
                img        = load_img(row)
                samples.append((img, row['label'], pred_label))

        if not samples:
            return

        n_cols = n_per_class
        n_rows = len(classes)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.5, n_rows * 3.5))
        fig.suptitle(f'{emoji}  {title}  ({len(samples)} / max {n_rows*n_cols})',
                     fontsize=15, fontweight='bold',
                     color=border_color, y=1.01)
        fig.patch.set_facecolor('#1a1a1a')

        axes = np.array(axes).reshape(n_rows, n_cols)

        sample_iter = iter(samples)
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r][c]
                ax.axis('off')
                try:
                    img, true_lbl, pred_lbl = next(sample_iter)
                    if img is not None:
                        ax.imshow(img)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(border_color)
                            spine.set_linewidth(3)
                        ax.set_title(
                            f'True: {true_lbl}\nPred: {pred_lbl}',
                            fontsize=8, color=border_color,
                            pad=4
                        )
                except StopIteration:
                    pass

        plt.tight_layout()
        if save_path:
            sp = save_path.replace('.png', f'_{title.lower()}.png')
            plt.savefig(sp, dpi=100, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f'저장: {sp}')
        plt.show()

    draw_grid('Success', correct_mask,   '#00e676', '✅')
    draw_grid('Fail',    incorrect_mask, '#ff1744', '❌')


sample_png = MODEL_PATH.replace('.pkl', '_samples.png')
show_sample_images(
    df_test, y_test, y_pred, le,
    data_dir=DATA_DIR,
    n_per_class=4,
    save_path=sample_png,
)