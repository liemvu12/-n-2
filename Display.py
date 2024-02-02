import cv2
import tensorflow as tf
import numpy as np
import os

filename = 'Dataset/DataFace/test/test (7).png'

# Lấy tên những người cần nhận diện
processed_data_path = 'Dataset/DataFace/processed'
folder_names = []

# Lặp qua các thư mục trong đường dẫn processed_data_path
for folder_name in os.listdir(processed_data_path):
    folder_path = os.path.join(processed_data_path, folder_name)

    # Kiểm tra xem folder_path là thư mục
    if os.path.isdir(folder_path):
        folder_names.append(folder_name)

# Nhận diện gương mặt với dữ liệu đã train
image = cv2.imread(filename)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
save_model = tf.keras.models.load_model("Khuon_mat.h5")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

fontface = cv2.FONT_HERSHEY_SIMPLEX

# Biến flag để kiểm tra có khuôn mặt nào được nhận diện hay không
face_detected = False

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Lấy vùng khuôn mặt
    roi_gray = gray[y:y+h, x:x+w]

    # Chuyển đổi kích thước vùng khuôn mặt
    roi_gray = cv2.resize(src=roi_gray, dsize=(100,100))

    # Biến đổi mảng thành dạng phù hợp cho model
    roi_gray = roi_gray.reshape((100, 100, 1))
    roi_gray = np.array(roi_gray)

    # Dự đoán khuôn mặt
    result = save_model.predict(np.array([roi_gray]))

    # Lấy kết quả dự đoán
    final = np.argmax(result)

    # Hiển thị tên người từ mảng folder_names
    if final < len(folder_names):
        person_name = folder_names[final]
        cv2.putText(image, person_name, (x+10, y+h+30), fontface, 1, (0, 255, 0), 2)
        face_detected = True

# Nếu không có khuôn mặt được nhận diện, hiển thị "không xác định"
if not face_detected:
    cv2.putText(image, "error", (10, 30), fontface, 1, (0, 0, 255), 2)

# Hiển thị ảnh
cv2.imshow('Khuon Mat', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
