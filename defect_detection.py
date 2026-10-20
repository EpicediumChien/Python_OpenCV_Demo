"""
OpenCV 瑕疵檢測範例

兩種模式：
1. 無參考模式（預設，不需要良品影像）
   頂帽運算 (Top-hat) 壓平背景 → SimpleBlobDetector 找圓形小斑點 → 面積/圓度/半徑過濾
2. 參考比對模式（指定 --reference 才啟用）
   ORB 特徵對齊 → 絕對差分 → 二值化 → 形態學去噪 → 連通區域過濾

兩種模式都會：依半徑門檻 (--max-radius) 篩選，保留的瑕疵以圓圈標出，
並輸出 Canny 邊緣圖、標註影像與 Markdown 報告。
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 避免 Windows 終端機（cp950/cp1252）輸出中文時出現編碼錯誤
sys.stdout.reconfigure(encoding="utf-8")

# 預設路徑
DEFAULT_IMAGE = Path(r"D:\GitHub\MBA_Lab1\Imgs\Final_Images\NG_1.bmp")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"

# 兩種模式各自的預設最小面積（像素）
DEFAULT_MIN_AREA_BLOB = 8
DEFAULT_MIN_AREA_REF = 80


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


# ---------------------------------------------------------------------------
# 無參考模式
# ---------------------------------------------------------------------------

def detect_blobs(gray: np.ndarray, tophat_size: int = 15, min_thresh: int = 15,
                 max_thresh: int = 60, min_area: int = 8, max_area: int = 150,
                 min_circularity: float = 0.7, min_convexity: float = 0.8,
                 max_radius: float | None = None):
    """
    在不需要參考影像的情況下找出「比周圍亮、尺寸小、形狀接近圓形」的斑點。

    1. 高斯模糊去噪
    2. 頂帽運算：把比結構元素大的背景（電路板大範圍亮度）去掉，只留下小於結構元素的亮點
    3. SimpleBlobDetector：依面積、圓度、凸度過濾，排除線路、焊墊等非圓形結構

    回傳:
        defects: 瑕疵列表 (dict，含 x, y, w, h, cx, cy, area, radius)
        tophat:  頂帽運算後的影像（可用來檢查偵測依據）
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_size, tophat_size))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255  # 偵測亮斑
    params.minThreshold = min_thresh
    params.maxThreshold = max_thresh
    params.thresholdStep = 5
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByConvexity = True
    params.minConvexity = min_convexity
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    keypoints = cv2.SimpleBlobDetector_create(params).detect(tophat)

    defects = []
    for kp in keypoints:
        cx, cy = kp.pt
        radius = kp.size / 2.0
        if max_radius is not None and radius >= max_radius:
            continue
        defects.append({
            "x": int(cx - radius), "y": int(cy - radius),
            "w": int(2 * radius), "h": int(2 * radius),
            "cx": float(cx), "cy": float(cy),
            "area": int(math.pi * radius * radius), "radius": float(radius),
        })

    defects.sort(key=lambda d: d["area"], reverse=True)
    return defects, tophat


# ---------------------------------------------------------------------------
# 參考比對模式（需要 --reference）
# ---------------------------------------------------------------------------

def align_to_reference(src: np.ndarray, ref: np.ndarray, min_inliers: int = 30):
    """
    以 ORB 特徵匹配 + 單應性矩陣，將 src 對齊到 ref。
    回傳 (對齊後影像, 有效區域遮罩, 內點數)；遮罩用來排除位移後產生的黑色空白邊緣。
    對齊失敗時回傳原影像、全白遮罩與內點數 0。
    """
    full_valid = np.full(ref.shape, 255, np.uint8)
    orb = cv2.ORB_create(5000)
    kp_src, des_src = orb.detectAndCompute(src, None)
    kp_ref, des_ref = orb.detectAndCompute(ref, None)
    if des_src is None or des_ref is None:
        print("[warn] 特徵點不足，略過對齊")
        return src, full_valid, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_src, des_ref), key=lambda m: m.distance)[:500]
    if len(matches) < 4:
        print("[warn] 匹配點不足，略過對齊")
        return src, full_valid, 0

    pts_src = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 3.0)
    if H is None or int(mask.sum()) < min_inliers:
        print("[warn] 對齊內點不足，略過對齊")
        return src, full_valid, 0

    inliers = int(mask.sum())
    print(f"[info] 對齊完成，內點數 {inliers}")
    size = (ref.shape[1], ref.shape[0])
    aligned = cv2.warpPerspective(src, H, size)
    valid = cv2.warpPerspective(np.full(src.shape, 255, np.uint8), H, size)
    # 向內縮 15 px，避開影像邊緣的插值與黑邊
    valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
    return aligned, valid, inliers


def detect_defects_ref(image: np.ndarray, reference: np.ndarray, valid: np.ndarray,
                       diff_thresh: int = 40, min_area: int = 80,
                       max_radius: float | None = None):
    """
    回傳:
        defects: 瑕疵列表 (dict，含 x, y, w, h, cx, cy, area, radius)
        diff:    差分灰階影像
        mask:    二值化瑕疵遮罩
    """
    img_blur = cv2.GaussianBlur(image, (5, 5), 0)
    ref_blur = cv2.GaussianBlur(reference, (5, 5), 0)

    diff = cv2.absdiff(img_blur, ref_blur)
    diff = cv2.bitwise_and(diff, valid)  # 無效區域（黑邊）不參與判斷

    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    defects = []
    for i in range(1, num):  # 0 為背景
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        # 以該連通區域的所有像素計算最小外接圓，取得圓心與半徑
        pts = cv2.findNonZero((labels == i).astype(np.uint8))
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        if max_radius is not None and radius >= max_radius:
            continue

        defects.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "cx": float(cx), "cy": float(cy),
            "area": int(area), "radius": float(radius),
        })

    defects.sort(key=lambda d: d["area"], reverse=True)
    return defects, diff, mask


# ---------------------------------------------------------------------------
# 共用
# ---------------------------------------------------------------------------

def detect_edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """對灰階影像做 Canny 邊緣偵測，回傳二值邊緣影像（邊緣為 255）。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, low, high)


def draw_result(color_img: np.ndarray, defects) -> np.ndarray:
    """以圓圈標出瑕疵（圓心與半徑取自偵測結果）並編號。"""
    out = color_img.copy()
    for idx, d in enumerate(defects, start=1):
        center = (int(round(d["cx"])), int(round(d["cy"])))
        r = int(round(d["radius"])) + 8  # 比瑕疵再大一圈，避免蓋住邊緣
        cv2.circle(out, center, r, (0, 0, 255), 3)
        cv2.putText(out, f"#{idx}", (center[0] - r, max(center[1] - r - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    status = "NG" if defects else "OK"
    color = (0, 0, 255) if defects else (0, 200, 0)
    cv2.putText(out, f"{status}: {len(defects)} defect(s)", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return out


def write_report(path: Path, args, method: str, params: list, shape,
                 inliers: int, defects, outputs: dict):
    """輸出 Markdown 報告，內容包含參數、判定結果、瑕疵表格與輸出檔說明。"""
    verdict = "NG" if defects else "OK"
    lines = [
        f"# 瑕疵檢測報告：{args.image.name}",
        "",
        f"- 執行時間：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 檢測模式：{method}",
        f"- 待檢影像：`{args.image}`",
        f"- 影像尺寸：{shape[1]} x {shape[0]}",
    ]
    if args.reference is not None:
        lines.append(f"- 參考影像：`{args.reference}`")
        lines.append(f"- 特徵對齊內點數：{inliers if inliers else '未對齊'}")
    lines += [
        "",
        "## 參數",
        "",
        "| 參數 | 值 |",
        "|------|----|",
    ]
    lines += [f"| {name} | {value} |" for name, value in params]
    lines += [
        "",
        "## 判定結果",
        "",
        f"**{verdict}**，共 {len(defects)} 個瑕疵",
        "",
    ]

    if defects:
        lines += [
            "| # | 圓心 X | 圓心 Y | 半徑 (px) | 面積 (px) |",
            "|---|--------|--------|-----------|-----------|",
        ]
        for idx, d in enumerate(defects, start=1):
            lines.append(
                f"| {idx} | {d['cx']:.0f} | {d['cy']:.0f} | "
                f"{d['radius']:.1f} | {d['area']} |"
            )
        lines.append("")

    lines += [
        "## 輸出檔案",
        "",
        "| 檔案 | 說明 |",
        "|------|------|",
    ]
    for name, desc in outputs.items():
        lines.append(f"| [{name}](./{name}) | {desc} |")
    lines.append("")

    lines += ["## 影像預覽", ""]
    for name in outputs:
        if name.endswith(".png"):
            lines += [f"![{name}](./{name})", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV 瑕疵檢測（無參考模式；指定 --reference 則改用參考比對）")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="待檢測影像")
    parser.add_argument("--reference", type=Path, default=None,
                        help="良品參考影像（有指定才使用參考比對模式）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="輸出資料夾")
    parser.add_argument("--min-area", type=int, default=None,
                        help="最小瑕疵面積（像素）；無參考模式預設 8，參考模式預設 80")
    parser.add_argument("--max-area", type=int, default=150,
                        help="最大斑點面積（像素，無參考模式）")
    parser.add_argument("--max-radius", type=float, default=None,
                        help="只保留半徑小於此值（像素）的瑕疵並以圓圈標出；未指定則全部標出")
    parser.add_argument("--tophat-size", type=int, default=15,
                        help="頂帽運算結構元素大小（無參考模式），需大於要找的小點")
    parser.add_argument("--min-circularity", type=float, default=0.7,
                        help="最小圓度 0-1（無參考模式），越高越只抓圓形")
    parser.add_argument("--diff-thresh", type=int, default=40,
                        help="差分二值化門檻（參考比對模式）")
    parser.add_argument("--no-align", action="store_true", help="不做特徵對齊（參考比對模式）")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny 低門檻")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny 高門檻")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    image = read_gray(args.image)
    color = read_color(args.image)
    print(f"[info] 待檢影像: {args.image} {image.shape}")

    # 邊緣偵測使用原始待檢影像（未對齊）
    edges = detect_edges(image, args.canny_low, args.canny_high)
    stem = args.image.stem
    outputs = {}
    params = [
        ("最大半徑 `--max-radius`",
         "不限" if args.max_radius is None else f"{args.max_radius} px（保留半徑小於此值）"),
        ("Canny 低/高門檻", f"{args.canny_low} / {args.canny_high}"),
    ]

    if args.reference is None:
        # ---- 無參考模式 ----
        method = "無參考模式（頂帽運算 + SimpleBlobDetector）"
        min_area = args.min_area if args.min_area is not None else DEFAULT_MIN_AREA_BLOB
        defects, tophat = detect_blobs(
            image, tophat_size=args.tophat_size, min_area=min_area,
            max_area=args.max_area, min_circularity=args.min_circularity,
            max_radius=args.max_radius,
        )
        inliers = 0
        params[0:0] = [
            ("頂帽結構元素 `--tophat-size`", f"{args.tophat_size} px"),
            ("最小面積 `--min-area`", f"{min_area} px"),
            ("最大面積 `--max-area`", f"{args.max_area} px"),
            ("最小圓度 `--min-circularity`", f"{args.min_circularity}"),
        ]
        cv2.imwrite(str(args.output / f"{stem}_tophat.png"), tophat)
        outputs[f"{stem}_tophat.png"] = "頂帽運算結果：亮度高於周圍的小點會呈現明亮"
    else:
        # ---- 參考比對模式 ----
        method = "參考比對模式（ORB 對齊 + 差分）"
        reference = read_gray(args.reference)
        print(f"[info] 參考影像: {args.reference} {reference.shape}")
        if image.shape != reference.shape:
            raise ValueError("待檢影像與參考影像尺寸不同")

        min_area = args.min_area if args.min_area is not None else DEFAULT_MIN_AREA_REF
        valid = np.full(reference.shape, 255, np.uint8)
        inliers = 0
        aligned = image
        if not args.no_align:
            aligned, valid, inliers = align_to_reference(image, reference)

        defects, diff, mask = detect_defects_ref(
            aligned, reference, valid, args.diff_thresh, min_area, args.max_radius
        )
        params[0:0] = [
            ("差分門檻 `--diff-thresh`", f"{args.diff_thresh}"),
            ("最小面積 `--min-area`", f"{min_area} px"),
        ]
        cv2.imwrite(str(args.output / f"{stem}_diff.png"), diff)
        cv2.imwrite(str(args.output / f"{stem}_mask.png"), mask)
        outputs[f"{stem}_diff.png"] = "與參考影像的灰階差分（越亮代表差異越大）"
        outputs[f"{stem}_mask.png"] = "瑕疵二值遮罩（白色為超過差分門檻的像素，已做形態學去噪）"

    result = draw_result(color, defects)
    cv2.imwrite(str(args.output / f"{stem}_edges.png"), edges)
    cv2.imwrite(str(args.output / f"{stem}_result.png"), result)

    outputs = {
        f"{stem}_result.png": "彩色標註影像：紅色圓圈為瑕疵，左上角為 OK/NG 判定",
        **outputs,
        f"{stem}_edges.png": "原始待檢影像的 Canny 邊緣偵測結果",
    }

    report_path = args.output / f"{stem}_report.md"
    write_report(report_path, args, method, params, image.shape, inliers, defects, outputs)

    print(f"[mode] {method}")
    print(f"[result] {'NG' if defects else 'OK'}，共 {len(defects)} 個瑕疵")
    for idx, d in enumerate(defects, start=1):
        print(f"  #{idx}: cx={d['cx']:.0f} cy={d['cy']:.0f} "
              f"area={d['area']} radius={d['radius']:.1f}")
    print(f"[info] 結果已儲存至 {args.output}")
    print(f"[info] 報告: {report_path}")


if __name__ == "__main__":
    main()
