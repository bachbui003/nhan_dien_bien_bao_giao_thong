import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# Danh sách tên biển báo
classes = {
    1: 'Giới hạn tốc độ (20km/h)',
    2: 'Giới hạn tốc độ (30km/h)',
    3: 'Giới hạn tốc độ (50km/h)',
    4: 'Giới hạn tốc độ (60km/h)',
    5: 'Giới hạn tốc độ (70km/h)',
    6: 'Giới hạn tốc độ (80km/h)',
    7: 'Hết giới hạn tốc độ (80km/h)',
    8: 'Giới hạn tốc độ (100km/h)',
    9: 'Giới hạn tốc độ (120km/h)',
    10: 'Không được vượt',
    11: 'Không được vượt xe có trọng tải trên 3,5 tấn',
    12: 'Quyền ưu tiên tại ngã tư',
    13: 'Đường ưu tiên',
    14: 'Nhường đường',
    15: 'Dừng lại',
    16: 'Không xe cộ',
    17: 'Cấm xe > 3,5 tấn',
    18: 'Không được vào',
    19: 'Cảnh báo chung',
    20: 'Rẽ trái nguy hiểm',
    21: 'Đường cong nguy hiểm bên phải',
    22: 'Đường cong đôi',
    23: 'Đường gập ghềnh',
    24: 'Đường trơn trượt',
    25: 'Đường hẹp bên phải',
    26: 'Công trình đường bộ',
    27: 'Tín hiệu giao thông',
    28: 'Người đi bộ',
    29: 'Trẻ em băng qua đường',
    30: 'Xe đạp băng qua đường',
    31: 'Cẩn thận băng/tuyết',
    32: 'Động vật hoang dã băng qua đường',
    33: 'Tốc độ kết thúc + giới hạn vượt',
    34: 'Rẽ phải phía trước',
    35: 'Rẽ trái phía trước',
    36: 'Chỉ đi phía trước',
    37: 'Đi thẳng hoặc rẽ phải',
    38: 'Đi thẳng hoặc rẽ trái',
    39: 'Giữ bên phải',
    40: 'Giữ bên trái',
    41: 'Bắt ​​buộc phải đi vòng',
    42: 'Kết thúc lệnh cấm vượt',
    43: 'Kết thúc lệnh cấm vượt xe > 3,5 tấn'
}

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.h5')

# Function to predict traffic sign from an image
def predict_image(img):
    # Convert to RGB if the image has RGBA channels
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize the image to 30x30 as expected by the model
    img = img.resize((30, 30))

    # Convert the image to a numpy array and normalize it
    img = np.array(img) / 255.0
    
    # Add batch dimension (model expects a batch of images)
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    
    # Get the predicted class name from the classes dictionary
    predicted_label = classes[predicted_class + 1]  # Since your classes start from 1, not 0

    return predicted_label, prediction[0][predicted_class] * 100  # return label and confidence

# Function to draw text on the image
def draw_text(img, text, position, font_size=30, color=(0, 255, 0)):
    # Tạo một đối tượng ImageDraw để vẽ văn bản
    draw = ImageDraw.Draw(img)
    
    # Chọn font chữ, bạn có thể thay đổi đường dẫn để chọn font hỗ trợ tiếng Việt
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Vẽ văn bản lên hình ảnh
    draw.text(position, text, font=font, fill=color)

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is captured correctly, ret is True
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to a PIL Image for processing
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Get the prediction
    predicted_label, confidence = predict_image(img_pil)
    
    # Vẽ văn bản tiếng Việt lên hình ảnh
    draw_text(img_pil, f"{predicted_label} ({confidence:.2f}%)", (10, 30))

    # Hiển thị hình ảnh
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow('Traffic Sign Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
