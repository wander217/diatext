# Cài Đặt
Để cài đặt thực hiện gõ lệnh sau
```
pip install -r requirements.txt
```
Link pretrained model:
```
"dbpp_se_eb3": "https://drive.google.com/file/d/1sAyndQTGH2YbX_XvBhA7scgmRE0A-mTm/view?usp=sharing"
```
# Class DBPredictor trong predictor.py
Dữ liệu khởi tạo bao gồm:
```
    config: lấy từ package config
    pretrainedPath: Đường dẫn tới pretrained
```
Gọi hàm predict với đầu vào một ảnh.
Dữ liệu trả về:
```
[
    {
        "bbox": Tọa độ của bounding box gồm 4 đỉnh dự đoán từ detection,
        "bbox_score": Độ tin cậy của bounding box
    },
]
```