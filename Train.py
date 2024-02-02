import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam

# Function to preprocess image
def preprocess_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Lỗi khi đọc ảnh: {image_path}")
        return None
    # Xử lý ảnh
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(src=frame, dsize=(100, 100))
    frame = np.array(frame)
    return frame

data = []
label = []
processed_data_path = 'Dataset/DataFace/processed'

# Biến đếm thư mục hiện tại
folder_idx = 0

for folder_name in os.listdir(processed_data_path):
    folder_path = os.path.join(processed_data_path, folder_name)

    # Kiểm tra xem folder_path là thư mục
    if os.path.isdir(folder_path):

        # Biến đếm file hiện tại
        idx = 0

        # Lặp qua các tệp ảnh trong thư mục con
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Kiểm tra xem tệp có phải là ảnh không
            if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    frame = preprocess_image(image_path)
                    if frame is not None:
                        data.append(frame)
                        label.append(folder_idx)
                        idx += 1
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh: {image_path}. Lỗi: {e}")

        # Tăng biến đếm cho vòng lặp thư mục
        folder_idx += 1

# Chuyển đổi danh sách thành mảng numpy
data = np.array(data)
label = np.array(label)
data = data.reshape((data.shape[0], 100, 100, 1))
X_train = data / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]

# One-hot encode nhãn
lb = LabelBinarizer()
Y_train = lb.fit_transform(label)

# Xây dựng mô hình CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(100, 100, 1)))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dense(len(np.unique(label))))  # Số lượng classes
model.add(Activation("softmax"))
model.summary()

# Compile và huấn luyện mô hình
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print("Bắt đầu huấn luyện")
model.fit(X_train, Y_train, batch_size=2, epochs=10)

# Lưu mô hình
model.save("Khuon_mat.h5")
