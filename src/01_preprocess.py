"""
01_preprocess.py
스쿼트 이미지에서 MediaPipe Pose로 관절 좌표를 뽑아 14개 피처를 CSV로 저장합니다.

- 원본 이미지 → features_orig.csv
- 원본 + 좌우 반전(flip) 증강 → features_aug.csv

사용법:
    python src/01_preprocess.py --data_dir data/images --out_dir data
"""
import os
import argparse

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

# ── visibility 임계값 ──
# 엄격한 임계값(CORE=0.5, AUX=0.3)에서는 측면 촬영 이미지의 83%가 탈락했습니다.
# 측면 촬영에서는 반대쪽 관절이 가려지는 것이 정상이라, 임계값을 크게 낮춰 데이터를 보존합니다.
CORE_KEYPOINTS = [11, 12, 23, 24, 25, 26, 27, 28]  # 어깨·골반·무릎·발목
MIN_VIS_CORE = 0.05

FEATURE_COLS = [
    'knee_L', 'knee_R', 'hip_L', 'hip_R',
    'ankle_L', 'ankle_R', 'cave_L', 'cave_R',
    'heel_lift_L', 'heel_lift_R',
    'torso_L', 'torso_R', 'sym_knee', 'knee_depth',
]

mp_pose = mp.solutions.pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.3,
)


# ============================================
# 피처 추출
# ============================================
def calc_angle(a, b, c):
    """세 점 a-b-c 에서 b를 꼭짓점으로 하는 각도(도)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def calc_vertical_angle(top, bottom):
    """두 점을 잇는 선분이 수직선과 이루는 각도(도)."""
    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]
    return np.degrees(np.arctan2(abs(dx), abs(dy)))


def extract_features_from_lm(lm):
    """MediaPipe 랜드마크 → 14개 피처. 거리값은 대퇴골 길이로 정규화합니다."""
    def pt(i): return [lm[i].x, lm[i].y]

    femur_L = np.sqrt((lm[23].x - lm[25].x) ** 2 + (lm[23].y - lm[25].y) ** 2)
    femur_R = np.sqrt((lm[24].x - lm[26].x) ** 2 + (lm[24].y - lm[26].y) ** 2)
    ref_length = (femur_L + femur_R) / 2 + 1e-6

    knee_L  = calc_angle(pt(23), pt(25), pt(27))
    knee_R  = calc_angle(pt(24), pt(26), pt(28))
    hip_L   = calc_angle(pt(11), pt(23), pt(25))
    hip_R   = calc_angle(pt(12), pt(24), pt(26))
    ankle_L = calc_angle(pt(25), pt(27), pt(29))
    ankle_R = calc_angle(pt(26), pt(28), pt(30))

    cave_L      = (lm[25].x - lm[27].x) / ref_length
    cave_R      = (lm[26].x - lm[28].x) / ref_length
    heel_lift_L = (lm[31].y - lm[29].y) / ref_length
    heel_lift_R = (lm[32].y - lm[30].y) / ref_length

    torso_L  = calc_vertical_angle(pt(11), pt(23))
    torso_R  = calc_vertical_angle(pt(12), pt(24))
    sym_knee = abs(knee_L - knee_R)

    knee_y_avg = (lm[25].y + lm[26].y) / 2
    hip_y_avg  = (lm[23].y + lm[24].y) / 2
    knee_depth = (knee_y_avg - hip_y_avg) / ref_length

    return [
        knee_L, knee_R, hip_L, hip_R, ankle_L, ankle_R,
        cave_L, cave_R, heel_lift_L, heel_lift_R,
        torso_L, torso_R, sym_knee, knee_depth,
    ]


def check_visibility(lm):
    return all(lm[i].visibility >= MIN_VIS_CORE for i in CORE_KEYPOINTS)


def process_array(img_bgr):
    result = mp_pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return None, 'no_pose'
    lm = result.pose_landmarks.landmark
    if not check_visibility(lm):
        return None, 'low_visibility'
    return extract_features_from_lm(lm), 'ok'


# ============================================
# 전체 순회
# ============================================
def main():
    parser = argparse.ArgumentParser(description='스쿼트 이미지 → 피처 CSV 변환')
    parser.add_argument('--data_dir', default=os.path.join(PROJECT_ROOT, 'data', 'images'),
                        help='클래스별 하위 폴더(bad_back / bad_heel / good)가 있는 이미지 폴더')
    parser.add_argument('--out_dir', default=os.path.join(PROJECT_ROOT, 'data'),
                        help='CSV 저장 폴더')
    args = parser.parse_args()

    rows_orig, rows_aug = [], []
    fail_log = {'no_pose': 0, 'low_visibility': 0, 'read_fail': 0}
    fail_flip = 0

    for label in sorted(os.listdir(args.data_dir)):
        folder = os.path.join(args.data_dir, label)
        if not os.path.isdir(folder):
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]
        print(f'\n[{label}] {len(files)}장 처리 중...')

        for fname in tqdm(files, desc=label):
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                fail_log['read_fail'] += 1
                continue

            feat, reason = process_array(img)
            if feat is None:
                fail_log[reason] += 1
            else:
                row = feat + [label, fname, 'orig']
                rows_orig.append(row)
                rows_aug.append(row)

            # 좌우 반전 증강
            feat_flip, _ = process_array(cv2.flip(img, 1))
            if feat_flip is None:
                fail_flip += 1
            else:
                rows_aug.append(feat_flip + [label, fname, 'flip'])

    columns = FEATURE_COLS + ['label', 'filename', 'aug']
    df_orig = pd.DataFrame(rows_orig, columns=columns)
    df_aug  = pd.DataFrame(rows_aug,  columns=columns)

    os.makedirs(args.out_dir, exist_ok=True)
    orig_csv = os.path.join(args.out_dir, 'features_orig.csv')
    aug_csv  = os.path.join(args.out_dir, 'features_aug.csv')
    df_orig.to_csv(orig_csv, index=False, encoding='utf-8-sig')
    df_aug.to_csv(aug_csv,  index=False, encoding='utf-8-sig')

    print('\n' + '=' * 50)
    print('전처리 완료')
    print(f'원본 CSV : {len(df_orig)}행 → {orig_csv}')
    print(f'증강 CSV : {len(df_aug)}행 → {aug_csv}')
    print('\n[실패 원인 분석]')
    print(f'포즈 미감지     : {fail_log["no_pose"]}장')
    print(f'visibility 미달 : {fail_log["low_visibility"]}장')
    print(f'파일 읽기 실패  : {fail_log["read_fail"]}장')
    print(f'flip 증강 실패  : {fail_flip}장')
    print('\n[원본 — 클래스 분포]')
    print(df_orig['label'].value_counts())
    print('\n[증강 — 클래스 분포]')
    print(df_aug['label'].value_counts())


if __name__ == '__main__':
    main()
