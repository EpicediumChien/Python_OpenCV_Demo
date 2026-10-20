"""
OpenCV 瑕疵檢測範例（參考影像比對法）

流程：
1. 讀取待檢測影像（灰階）與良品參考影像 reference.bmp
2. 用 ORB 特徵 + RANSAC 單應性矩陣把待檢影像對齊到參考影像（抵消輕微位移/旋轉）
3. 做絕對差分 (absdiff)，找出與良品不同的區域
4. 二值化 + 形態學運算去除雜訊
5. 找連通區域，依面積過濾，框出瑕疵位置並輸出結果
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 避免 Windows 終端機（cp950/cp1252）輸出中文時出現編碼錯誤
sys.stdout.reconfigure(encoding="utf-8")

# 預設路徑
DEFAULT_IMAGE = Path(r"D:\GitHub\MBA_Lab1\Imgs\Final_Images\NG_1.bmp")
DEFAULT_REFERENCE = Path(r"D:\GitHub\MBA_Lab1\Imgs\Final_Images\reference.bmp")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


def read_gray(path: Path) -> np.ndarray:
    """讀取灰階影像（使用 imdecode 以支援含非 ASCII 字元的路徑）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"無法讀取影像: {path}")
    return img


def read_color(path: Path) -> np.ndarray:
    """讀取彩色影像，用於輸出標註結果。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"無法讀取影像: {path}")
    return img


def align_to_reference(src: np.ndarray, ref: np.ndarray, min_inliers: int = 30):
    """
    以 ORB 特徵匹配 + 單應性矩陣，將 src 對齊到 ref。
    回傳 (對齊後影像, 有效區域遮罩)；遮罩用來排除位移後產生的黑色空白邊緣。
    對齊失敗時回傳原影像與全白遮罩。
    """
    full_valid = np.full(ref.shape, 255, np.uint8)
    orb = cv2.ORB_create(5000)
    kp_src, des_src = orb.detectAndCompute(src, None)
    kp_ref, des_ref = orb.detectAndCompute(ref, None)
    if des_src is None or des_ref is None:
        print("[warn] 特徵點不足，略過對齊")
        return src, full_valid

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_src, des_ref), key=lambda m: m.distance)[:500]
    if len(matches) < 4:
        print("[warn] 匹配點不足，略過對齊")
        return src, full_valid

    pts_src = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 3.0)
    if H is None or int(mask.sum()) < min_inliers:
        print("[warn] 對齊內點不足，略過對齊")
        return src, full_valid

    print(f"[info] 對齊完成，內點數 {int(mask.sum())}")
    size = (ref.shape[1], ref.shape[0])
    aligned = cv2.warpPerspective(src, H, size)
    valid = cv2.warpPerspective(np.full(src.shape, 255, np.uint8), H, size)
    # 向內縮 15 px，避開影像邊緣的插值與黑邊
    valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
    return aligned, valid


def detect_defects(image: np.ndarray, reference: np.ndarray, valid: np.ndarray,
                   diff_thresh: int = 40, min_area: int = 80):
    """
    回傳:
        boxes: 瑕疵外接框 [(x, y, w, h, area), ...]
        diff:  差分灰階影像
        mask:  二值化瑕疵遮罩
    """
    # 1. 高斯模糊抑制感測器雜訊
    img_blur = cv2.GaussianBlur(image, (5, 5), 0)
    ref_blur = cv2.GaussianBlur(reference, (5, 5), 0)

    # 2. 絕對差分：與良品不同的地方亮度值會變大
    diff = cv2.absdiff(img_blur, ref_blur)
    diff = cv2.bitwise_and(diff, valid)  # 無效區域（黑邊）不參與判斷

    # 3. 二值化
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    # 4. 形態學：開運算去除小雜點，閉運算填補區域縫隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. 連通區域分析，依面積過濾
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, num):  # 0 為背景
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h), int(area)))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes, diff, mask


def draw_result(color_img: np.ndarray, boxes) -> np.ndarray:
    """在影像上畫出瑕疵框與編號。"""
    out = color_img.copy()
    for idx, (x, y, w, h, area) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(out, f"#{idx}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    status = "NG" if boxes else "OK"
    color = (0, 0, 255) if boxes else (0, 200, 0)
    cv2.putText(out, f"{status}: {len(boxes)} defect(s)", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return out


def main():
    parser = argparse.ArgumentParser(description="OpenCV 瑕疵檢測（參考影像比對）")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="待檢測影像")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="良品參考影像")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="輸出資料夾")
    parser.add_argument("--diff-thresh", type=int, default=40, help="差分二值化門檻 (0-255)")
    parser.add_argument("--min-area", type=int, default=80, help="最小瑕疵面積（像素）")
    parser.add_argument("--no-align", action="store_true", help="不做特徵對齊")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    image = read_gray(args.image)
    reference = read_gray(args.reference)
    color = read_color(args.image)
    print(f"[info] 待檢影像: {args.image} {image.shape}")
    print(f"[info] 參考影像: {args.reference} {reference.shape}")

    if image.shape != reference.shape:
        raise ValueError("待檢影像與參考影像尺寸不同")

    valid = np.full(reference.shape, 255, np.uint8)
    if not args.no_align:
        image, valid = align_to_reference(image, reference)

    boxes, diff, mask = detect_defects(image, reference, valid, args.diff_thresh, args.min_area)

    result = draw_result(color, boxes)

    stem = args.image.stem
    cv2.imwrite(str(args.output / f"{stem}_result.png"), result)
    cv2.imwrite(str(args.output / f"{stem}_diff.png"), diff)
    cv2.imwrite(str(args.output / f"{stem}_mask.png"), mask)

    print(f"[result] {'NG' if boxes else 'OK'}，共 {len(boxes)} 個瑕疵")
    for idx, (x, y, w, h, area) in enumerate(boxes, start=1):
        print(f"  #{idx}: x={x} y={y} w={w} h={h} area={area}")
    print(f"[info] 結果已儲存至 {args.output}")


if __name__ == "__main__":
    main()
