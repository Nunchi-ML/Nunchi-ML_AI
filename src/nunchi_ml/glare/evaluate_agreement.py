"""성진 님 라벨과 범수 님 독립 라벨 간의 2인 일치도(코헨 카파)를 평가하고
불일치 사진 목록 및 합의 라벨을 생성하는 모듈.

사용:
    python -m src.glare.evaluate_agreement data/glare_pilot_labels.csv data/glare_labeling_sheet_범수.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

sys.stdout.reconfigure(encoding="utf-8")

LABEL_NAMES = {0: "0 정상", 1: "1 눈부심", 2: "2 심한 눈부심"}


def main(orig_csv, my_csv):
    df_orig = pd.read_csv(orig_csv)
    df_my = pd.read_csv(my_csv)

    # image_id 또는 (vid, file) 기준 병합
    if "image_id" in df_my.columns and "pid" in df_orig.columns:
        m = df_orig.merge(df_my, left_on="pid", right_on="image_id", suffixes=("_성진", "_범수"))
    else:
        m = df_orig.merge(df_my, on=["vid", "file"], suffixes=("_성진", "_범수"))

    # 라벨 유효성 검사
    m["label_범수"] = pd.to_numeric(m["label_범수"], errors="coerce")
    valid_mask = m["label_범수"].notna()
    n_labeled = valid_mask.sum()
    print(f"==================================================")
    print(f"👥 2인 독립 라벨링 일치도 분석 (총 {len(m)}장 중 {n_labeled}장 라벨링 완료)")
    print(f"==================================================")

    if n_labeled == 0:
        print("[안내] 아직 'glare_labeling_sheet_범수.csv'에 라벨(0, 1, 2)이 입력되지 않았습니다.")
        print("라벨시트 이미지를 확인하고 라벨을 채워 넣은 뒤 다시 실행해 주세요.")
        return

    sub = m[valid_mask].copy()
    y_orig = sub["label_성진"].astype(int).values
    y_my = sub["label_범수"].astype(int).values

    # 1. 일치율 및 혼동행렬
    acc = accuracy_score(y_orig, y_my)
    cm = confusion_matrix(y_orig, y_my, labels=[0, 1, 2])
    print(f"\n1. 전체 단순 일치율: {acc * 100:.1f}% ({acc:.3f})")
    print("\n[혼동행렬 (행 = 성진, 열 = 범수)]")
    cm_df = pd.DataFrame(cm, index=["성진: 0", "성진: 1", "성진: 2"], columns=["범수: 0", "범수: 1", "범수: 2"])
    print(cm_df)

    # 클래스별 일치율 (Class-wise Specific Agreement: 2*A / (N1 + N2))
    print("\n[클래스별 일치율 (Specific Agreement)]")
    for k in [0, 1, 2]:
        agree_k = 2 * cm[k, k] / (cm[k, :].sum() + cm[:, k].sum()) if (cm[k, :].sum() + cm[:, k].sum()) > 0 else 0
        iou_k = cm[k, k] / (cm[k, :].sum() + cm[:, k].sum() - cm[k, k]) if (cm[k, :].sum() + cm[:, k].sum() - cm[k, k]) > 0 else 0
        print(f"  - {LABEL_NAMES[k]}: 특정 일치율 {agree_k * 100:.1f}%, IoU 일치율 {iou_k * 100:.1f}% (둘 다 일치: {cm[k, k]}건)")

    # 2. 가중 코헨 카파 (순서형 3단계)
    kappa_linear = cohen_kappa_score(y_orig, y_my, weights="linear")
    kappa_quad = cohen_kappa_score(y_orig, y_my, weights="quadratic")
    kappa_unweighted = cohen_kappa_score(y_orig, y_my)

    print(f"\n2. 코헨 카파 계수 (Cohen's Kappa):")
    print(f"  - 단순 비가중 카파 (Unweighted): {kappa_unweighted:.3f}")
    print(f"  - 선형 가중 카파 (Linear Weighted)   : {kappa_linear:.3f}")
    print(f"  - 이차 가중 카파 (Quadratic Weighted): {kappa_quad:.3f}")

    def kappa_grade(k):
        if k >= 0.81: return "거의 완전한 일치 (Almost Perfect)"
        if k >= 0.61: return "상당한 일치 (Substantial)"
        if k >= 0.41: return "적당한 일치 (Moderate)"
        if k >= 0.21: return "약한 일치 (Fair)"
        return "미미한 일치 (Slight)"

    print(f"  👉 평가: {kappa_grade(kappa_quad)}")

    # 3. 불일치 사진 목록 (회의 안건)
    diff_mask = y_orig != y_my
    diff_df = sub[diff_mask].copy()
    print(f"\n3. 불일치 사진 (총 {len(diff_df)}장): 회의 합의 대상")
    cols_show = ["image_id", "sheet_name", "sheet_location", "video_id", "label_성진", "label_범수"]
    cols_exist = [c for c in cols_show if c in diff_df.columns]
    print(diff_df[cols_exist].to_string(index=False))

    # 4. 잠정 합의 라벨 생성 (기준: "경계가 모호하면 낮은 등급, 단 위험 요소 시 Class 2 검토")
    # 원칙:
    # - 성진과 범수 일치 -> 그대로 확정
    # - 불일치 시: 도로 윤곽/차선/표지판 가독성이 심각하게 저하되어 한 평가자가 2를 준 경우, 안전 보수적 관점에서 2 반영 검토
    #   (단 0 vs 1 경계는 보수적으로 0 또는 1 협의)
    # 1차 자동 합의: 계획서 원칙 "일치분 우선, 불일치는 안전 위험 관점에서 더 높은 등급(위험 우선) 또는 협의"
    # 여기서는 계획서에 명시된 원칙 "도로·차선·전방차량·표지판 중 하나라도 실질적 식별 불가 시 Class 2 우선 검토"를 적용하여
    # 1 vs 2 불일치는 2(심한 눈부심)로, 0 vs 1 불일치는 보수적으로 0(정상)으로 1차 합의 처리
    consensus = []
    for s_l, b_l in zip(sub["label_성진"], sub["label_범수"]):
        if s_l == b_l:
            consensus.append(s_l)
        elif {s_l, b_l} == {1, 2}:
            consensus.append(2)  # 심한 눈부심 위험 우선 반영
        elif {s_l, b_l} == {0, 1}:
            consensus.append(0)  # 보수적으로 낮은 등급
        else:
            consensus.append(min(s_l, b_l))
    sub["label_consensus"] = consensus

    file_col = "file_성진" if "file_성진" in sub.columns else ("file" if "file" in sub.columns else "file_범수")
    vid_col = "vid" if "vid" in sub.columns else "video_id"
    
    out_consensus = "data/glare_consensus_labels.csv"
    res_df = pd.DataFrame({
        "vid": sub[vid_col],
        "file": sub[file_col],
        "label": sub["label_consensus"]
    })
    res_df.to_csv(out_consensus, index=False, encoding="utf-8-sig")
    print(f"\n✅ 잠정 합의 라벨 저장 완료: {out_consensus}")
    print(f"  합의 라벨 분포: 정상 {sum(res_df['label']==0)} / 눈부심 {sum(res_df['label']==1)} / 심한 {sum(res_df['label']==2)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
