from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Eğitilmiş modeli yükle
model=load_model("Models/MyModel01.h5")

# MNIST veri kümesini yükle
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# Test verilerini hazırla
x_test=x_test.reshape(10000,28,28,1)
x_test=x_test.astype("float32")/255

# Palet ve ekran boyutunu ayarlayın
palette_size = 28
display_size = 280

# Oranları hesaplayın
ratio = display_size / palette_size

# Boş bir tuval oluştur
canvas = np.zeros((28, 28), dtype=np.uint8)

# Çizim penceresini oluştur
cv2.namedWindow("Drawing Canvas")

prev_point = None

# Çizim işlevi tanımla
def draw(event, x, y, flags, param):
    global canvas, prev_point

    if event == cv2.EVENT_LBUTTONDOWN:
        prev_point = (int(x / ratio), int(y / ratio))

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if prev_point is not None:
            cv2.line(canvas, prev_point, (int(x / ratio), int(y / ratio)), 255, 2)
            prev_point = (int(x / ratio), int(y / ratio))

    elif event == cv2.EVENT_LBUTTONUP:
        prev_point = None

# Fare olaylarını takip etmek üzere işlevi belirt
cv2.setMouseCallback("Drawing Canvas", draw)

while True:
    # Tuvali yeniden boyutlandır ve göster
    display_canvas = cv2.resize(canvas, (280, 280))
    cv2.imshow("Drawing Canvas", display_canvas)
    
    # Tuşa basma olaylarını takip et
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Çıkış yap
        break
    elif key == ord('c'): # Tuvali temizle
        canvas.fill(0)
    elif key == ord('s'): # Çizim verisini modelin giriş formatına dönüştür ve tahminle
        if np.max(canvas) > 0: 
            resized_canvas = cv2.resize(canvas, (28, 28))
            input_data = resized_canvas.reshape(1, 28, 28, 1).astype("float32") / 255
            prediction = np.argmax(model.predict(input_data))
            print("Prediction Label:", prediction)

cv2.destroyAllWindows()



    



















