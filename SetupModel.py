from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Dropout,BatchNormalization,MaxPooling2D

# MNIST veri kümesi yüklenir
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# Veri şekillendirilir ve normalleştirilir
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)

x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

# Etiketler kategorik hale getirilir
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_train,10)

# Eğitim ve doğrulama verileri oluşturulur
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.2,random_state=0)

# Evrişimli sinir ağı modeli oluşturulur
model = Sequential()

# İlk evrişim katmanı
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"))
model.add(BatchNormalization()) # Normalleştirme

# İkinci evrişim katmanı
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # Maksimum havuzlama
model.add(Dropout(0.25)) # Dropout , aşırı örenmeyi engellemek için

# Üçüncü evrişim katmanı
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))


# Dördüncü evrişim katmanı
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


# Düzleştirme katmanı
model.add(Flatten())

# Tam bağlı (fully connected) katmanlar
model.add(Dense(512,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Çıkış katmanı
model.add(Dense(10,activation="softmax"))

# Modeli derleme
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Model eğitimi gerçekleştirilir
output = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=35, batch_size=128, verbose=0)

# Eğitilen model kaydedilir
model.save("Models/MyModel01.h5")

# Eğitim sürecinin loss ve accuracy grafiği çizilir
fig, ax = plt.subplots(1, 2)

ax[0].plot(output.history["loss"], label="Training Loss")
ax[0].plot(output.history["val_loss"], label="Validation Loss")
ax[0].set_title("Loss Graph")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend()

ax[1].plot(output.history["accuracy"], label="Training Accuracy")
ax[1].plot(output.history["val_accuracy"], label="Validation Accuracy")
ax[1].set_title("Accuracy Graph")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

plt.show()