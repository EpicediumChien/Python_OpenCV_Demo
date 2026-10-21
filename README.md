# Python OpenCV 瑕疵檢測範例

在 PCB 影像上找出比周圍亮的小瑕疵，並描出每個瑕疵的實際輪廓。

需要 Python 3.10 以上。

## 安裝與執行

```bash
pip install -r requirements.txt
python defect_detection.py                      # 預設檢測同資料夾的 NG_1.bmp
python defect_detection.py --image OK_1.bmp --thresh 40
python defect_detection.py --exclude 0,350,420,450 --exclude 540,320,180,160
```

## 兩種模式

**無參考模式（預設）**：不需要良品影像。

1. 高斯模糊去噪
2. 頂帽運算（Top-hat）壓平大範圍背景，只留下比結構元素小的亮點
3. 以 `--thresh` 二值化，取外輪廓
4. 依面積、圓度、凸度過濾，排除線路與焊墊等非圓形結構
5. 描出每個瑕疵的輪廓並編號，輸出 OK／NG 與報告

**參考比對模式**（指定 `--reference`）：ORB 特徵＋RANSAC 對齊 → 絕對差分 → 二值化 → 形態學去噪 → 連通區域過濾，同樣描出輪廓。此模式需要良品參考影像，目前尚未用實際影像驗證。

## 參數

| 參數 | 說明 | 預設 |
|------|------|------|
| `--image` | 待檢測影像 | `NG_1.bmp` |
| `--reference` | 良品參考影像（指定才啟用參考比對模式） | 無 |
| `--output` | 結果輸出資料夾 | `output/` |
| `--thresh` | 頂帽影像二值化門檻 0–255，越小越敏感（無參考模式） | 30 |
| `--tophat-size` | 頂帽結構元素大小，需大於要找的小點 | 15 |
| `--min-area` / `--max-area` | 瑕疵面積範圍（像素） | 8 / 150 |
| `--min-circularity` | 最小圓度 0–1，越高越只抓圓形 | 0.7 |
| `--max-radius` | 只保留外接圓半徑小於此值的瑕疵 | 不限 |
| `--exclude X,Y,W,H` | 不檢測的矩形區域，可重複指定 | 無 |
| `--diff-thresh` | 差分二值化門檻（參考比對模式） | 40 |
| `--canny-low` / `--canny-high` | Canny 邊緣圖門檻（僅輸出參考，不參與判定） | 50 / 150 |

## 輸出（`output/`）

- `<name>_result.png`：紅色輪廓標出瑕疵，左上角為 OK／NG 判定
- `<name>_tophat.png`：頂帽運算結果
- `<name>_edges.png`：Canny 邊緣圖
- `<name>_report.md`：參數、判定結果與瑕疵表（中心、面積、周長、圓度）
