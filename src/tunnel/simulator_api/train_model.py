"""대표 모델 학습 → model3_대표모델.joblib 저장 + 10 노트북 결과와 대조.
실행: python train_model.py --train <학습표_71786.csv> [--test <시험표_mapillary.csv> --ref <등급2단계_사진별.csv>]
학습 데이터 = AI Hub 71786 3,624장 (2023년 8~9월, 입구 1,927 + 터널 없는 고속도로 1,697). 시험 Mapillary는 학습에 안 씀."""
import argparse, joblib, numpy as np, pandas as pd, sklearn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, r2_score
from model3_core import PLUS, MAIN_COLS, LAB, ratio_from_L20, grade

C_MAIN = 0.01          # 11/6 형 확정값 (10 노트북에서 학습 OOF F1로 고른 값)
ridge = lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10))
logit = lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              LogisticRegression(C=C_MAIN, class_weight='balanced', max_iter=2000))


def fit(tr):
    gtr = grade(ratio_from_L20(10 ** tr.y.values, tr.cls.values))
    est = np.empty(len(tr))                 # 학습 사진의 추정 밝기 = 장면 5겹 OOF (자기 정답 안 봄)
    for a, b in GroupKFold(5).split(tr, groups=tr.scene.values):
        est[b] = ridge().fit(tr.iloc[a][PLUS], tr.y.iloc[a]).predict(tr.iloc[b][PLUS])
    t2 = tr.copy(); t2['est_L'] = est
    r = ridge().fit(tr[PLUS], tr.y)          # 새 사진 밝기 추정 = 학습 전체로 학습
    X = t2[MAIN_COLS].to_numpy(float); risk = gtr != '하'
    m1 = logit().fit(X, risk)                # 1단계: 위험(중·상)인가
    m2 = logit().fit(X[risk], gtr[risk] == '상')   # 2단계: (위험만) 상인가
    return {'ridge': r, 'stage1': m1, 'stage2': m2, 'PLUS': PLUS, 'MAIN_COLS': MAIN_COLS, 'C': C_MAIN,
            'sklearn': sklearn.__version__, 'n_train': len(tr)}


def predict_df(M, d):
    d = d.copy(); d['est_L'] = M['ridge'].predict(d[M['PLUS']])
    X = d[M['MAIN_COLS']].to_numpy(float)
    p1 = M['stage1'].predict_proba(X)[:, 1]; p2 = M['stage2'].predict_proba(X)[:, 1]
    r1 = M['stage1'].predict(X).astype(bool); r2 = M['stage2'].predict(X).astype(bool)
    d['등급'] = np.where(~r1, '하', np.where(r2, '상', '중'))
    d['p_위험'] = p1; d['p_상_위험중'] = p2
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--test'); ap.add_argument('--ref')
    ap.add_argument('--out', default='model3_대표모델.joblib'); a = ap.parse_args()
    tr = pd.read_csv(a.train); M = fit(tr); joblib.dump(M, a.out); print('저장', a.out, '| 학습', len(tr), '장')
    if a.test:
        mp = pd.read_csv(a.test, dtype={'image_id': str})
        te = mp[mp.test_ok & (mp.cls == '입구')].reset_index(drop=True)
        P = predict_df(M, te); g = grade(ratio_from_L20(10 ** te.y.values, te.cls.values))
        print('입구', len(te), '장 | Macro F1', round(f1_score(g, P['등급'], average='macro', labels=LAB), 3),
              '| 상 재현', round(((P['등급'] == '상') & (g == '상')).sum() / (g == '상').sum(), 2),
              '| 밝기 R²', round(r2_score(te.y, P.est_L), 3))
        if a.ref:
            ref = pd.read_csv(a.ref, dtype={'image_id': str})
            col = [c for c in ref.columns if c.startswith('주 모델')][0]
            m = P[['image_id', '등급', 'est_L']].merge(ref[['image_id', col, '추정 L20 (주 모델)']], on='image_id')
            print('10 노트북과 등급 일치', int((m['등급'] == m[col]).sum()), '/', len(m),
                  '| 추정 L20 최대 차이', round(float(np.abs(10 ** m.est_L - m['추정 L20 (주 모델)']).max()), 2), 'cd/m²')
