import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
import os

# Load the trained model to classify signs
model = load_model('my_model.h5')

# Dictionary to label all traffic signs class.
classes = {1:'Giới hạn tốc độ (20km/h)',
2:'Giới hạn tốc độ (30km/h)',
3:'Giới hạn tốc độ (50km/h)',
4:'Giới hạn tốc độ (60km/h)',
5:'Giới hạn tốc độ (70km/h)',
6:'Giới hạn tốc độ (80km/h)',
7:'Hết giới hạn tốc độ (80km/h)',
8:'Giới hạn tốc độ (100km/h)',
9:'Giới hạn tốc độ (120km/h)',
10:'Không được vượt',
11:'Không được vượt xe có trọng tải trên 3,5 tấn',
12:'Quyền ưu tiên tại ngã tư',
13:'Đường ưu tiên',
14:'Nhường đường',
15:'Dừng lại',
16:'Không xe cộ', 
17:'Cấm xe > 3,5 tấn', 
18:'Không được vào', 
19:'Cảnh báo chung', 
20:'Rẽ trái nguy hiểm', 
21:'Đường cong nguy hiểm bên phải', 
22:'Đường cong đôi', 
23:'Đường gập ghềnh', 
24:'Đường trơn trượt', 
25:'Đường hẹp bên phải', 
26:'Công trình đường bộ', 
27:'Tín hiệu giao thông', 
28:'Người đi bộ', 
29:'Trẻ em băng qua đường', 
30:'Xe đạp băng qua đường', 
31:'Cẩn thận băng/tuyết', 
32:'Động vật hoang dã băng qua đường', 
33:'Tốc độ kết thúc + giới hạn vượt', 
34:'Rẽ phải phía trước', 
35:'Rẽ trái phía trước', 
36:'Chỉ đi phía trước', 
37:'Đi thẳng hoặc rẽ phải', 
38:'Đi thẳng hoặc rẽ trái', 
39: 'Giữ bên phải', 
40: 'Giữ bên trái', 
41: 'Bắt ​​buộc phải đi vòng', 
42: 'Kết thúc lệnh cấm vượt', 
43: 'Kết thúc lệnh cấm vượt xe > 3,5 tấn' }

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Nhận diện biển báo giao thông')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((30, 30))  # Resize image to the size expected by the model
    image = np.array(image) / 255.0  # Normalize the pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Predict the class
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the class with highest probability
    sign = classes[predicted_class + 1]  # Get the name of the class (1-based index)
    
    print(sign)  # Print the predicted class
    label.configure(foreground='#011638', text=sign)  # Update label with predicted sign

def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận diện", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')  # Clear the previous label text
        show_classify_button(file_path)
    except Exception as e:
        print(e)

# Function to run webcam.py when the "Nhận diện bằng webcam" button is pressed
def run_webcam():
    os.system("python webcam.py")

# Create and place the "Nhận diện bằng webcam" button on the left
webcam_button = Button(top, text="Nhận diện bằng webcam", command=run_webcam, padx=10, pady=5)
webcam_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
webcam_button.grid(row=0, column=0, padx=5)

# Create and place the "Chọn ảnh" button on the right
upload_button = Button(top, text="Chọn ảnh", command=upload_image, padx=10, pady=5)
upload_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload_button.grid(row=0, column=1, padx=5)

# Header label
heading = Label(top, text="Nhận diện biển báo", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.grid(row=1, column=0, columnspan=2)

# Place the image and result label at the bottom
sign_image.grid(row=2, column=0, columnspan=2, pady=20)
label.grid(row=3, column=0, columnspan=2, pady=20)

# Run the Tkinter main loop
top.mainloop()
