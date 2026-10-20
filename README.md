# Python OpenCV 瑕疵檢測範例

以良品參考影像比對的方式，找出待檢 PCB 影像中的瑕疵。

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

預設檢測 `D:\GitHub\MBA_Lab1\Imgs\Final_Images\NG_1.bmp`，以 `reference.bmp` 為良品參考：

```bash
python defect_detection.py
```

指定其他影像與參數：

```bash
python defect_detection.py --image OK_1.bmp --diff-thresh 40 --min-area 80
```

| 參數 | 說明 | 預設 |
|------|------|------|
| `--image` | 待檢測影像 | NG_1.bmp |
| `--reference` | 良品參考影像 | reference.bmp |
| `--output` | 結果輸出資料夾 | `output/` |
| `--diff-thresh` | 差分二值化門檻（0–255），越小越敏感 | 40 |
| `--min-area` | 最小瑕疵面積（像素），用來濾除雜訊 | 80 |
| `--no-align` | 不做特徵對齊（影像未位移時可用） | 關閉 |

## 流程

1. 讀取灰階待檢影像與參考影像
2. ORB 特徵 + RANSAC 單應性矩陣，將待檢影像對齊到參考影像
3. 高斯模糊後做絕對差分，排除對齊後的黑色空白邊緣
4. 二值化、形態學開/閉運算去除雜訊
5. 連通區域分析，依面積過濾，框出瑕疵並輸出結果

## 輸出

輸出至 `output/`：

- `<name>_result.png`：標註紅框與 OK/NG 判定的彩色影像
- `<name>_diff.png`：差分灰階影像
- `<name>_mask.png`：瑕疵二值遮罩
