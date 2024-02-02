import numpy as np
import cv2
import os

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đường dẫn thư mục raw và processed
raw_data_path = 'Dataset/DataFace/raw'
processed_data_path = 'Dataset/DataFace/processed'

# Lặp qua các thư mục trong raw
for folder_name in os.listdir(raw_data_path):
    folder_path = os.path.join(raw_data_path, folder_name)

    # Kiểm tra xem folder_path là thư mục
    if os.path.isdir(folder_path):
        # Tạo đường dẫn cho thư mục processed tương ứng
        processed_folder_path = os.path.join(processed_data_path, folder_name)

        # Lặp qua các tệp ảnh trong thư mục con
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Kiểm tra xem tệp có phải là ảnh không
            if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    frame = cv2.imread(image_path)
                    if frame is None:
                        print(f"Error reading image: {image_path}")
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Phát hiện khuôn mặt
                    faces = detector.detectMultiScale(gray, 1.1, 5)

                    # Tạo thư mục mới nếu chưa tồn tại
                    os.makedirs(processed_folder_path, exist_ok=True)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        processed_file_path = os.path.join(processed_folder_path, image_name)

                        # Lưu ảnh đã xử lý vào thư mục processed
                        cv2.imwrite(processed_file_path, gray[y:y + h, x:x + w])

                except Exception as e:
                    print(f"Error processing image: {image_path}")
                    print(e)
