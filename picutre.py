import os
import numpy as np
import pandas as pd

from PIL import Image  # or opencv if you prefer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

##############################################################################
# 1) LOAD NUMERIC DATA FROM CSV
##############################################################################
def load_numeric_data(csv_file):
    """
    Assume the CSV has columns like:
      date, open, high, low, close, indicator1, indicator2, ..., y
    where y is the 0/1 label for bullish/bearish.
    We'll drop anything else or fill as needed.
    """
    df = pd.read_csv(csv_file)
    # Ensure 'y' is present
    if 'y' not in df.columns:
        raise ValueError("CSV must have a 'y' column for the label (0 or 1).")

    # Example: drop 'date' if not needed. Or keep if you want.
    # We'll just do a minimal approach
    # Let's assume the rest are numeric features except 'y'.
    # E.g. columns = [date, open, high, low, close, stoch, RSI, y]
    # We'll do:
    features = [c for c in df.columns if c not in ['date','y']]
    
    X_num = df[features].to_numpy()
    y = df['y'].to_numpy()

    return X_num, y

##############################################################################
# 2) LOAD SCREENSHOTS (IMAGES) FROM FOLDER
##############################################################################
def load_images(folder_path, num_samples, img_size=(64,64)):
    """
    We'll assume images are named 'img_0.png', 'img_1.png', ... 'img_{num_samples-1}.png'
    and each file corresponds to the same row in the CSV by index.
    
    We return an array shape: (num_samples, 64, 64, 3) for a color image (example).
    """
    X_images = []
    for i in range(num_samples):
        # build filename
        fn = os.path.join(folder_path, f"img_{i}.png")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Missing image file: {fn}")
        # load image
        img = Image.open(fn).convert('RGB')  # ensure 3 channels
        img = img.resize(img_size)           # resize to 64x64
        arr = np.array(img)                  # shape (64,64,3)
        X_images.append(arr)
    
    X_images = np.array(X_images, dtype=np.float32)
    # optionally scale pixel intensities to [0,1]
    X_images /= 255.0
    return X_images

##############################################################################
# 3) BUILD A SIMPLE CNN FOR IMAGE CLASSIFICATION
##############################################################################
def build_cnn_model(input_shape=(64,64,3)):
    model = Sequential()
    # small, simple CNN
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

##############################################################################
# 4) MAIN WORKFLOW - LATE FUSION
##############################################################################
def main():
    csv_file = "eurusd_data.csv"
    folder_images = "screenshots/"
    
    # 1) load numeric data
    X_num, y = load_numeric_data(csv_file)
    num_samples = len(X_num)
    
    # 2) load images
    X_img = load_images(folder_images, num_samples, img_size=(64,64))
    
    # 3) train/test split
    # We'll do a single random split for demonstration
    # (some prefer a time-based split for forex)
    X_train_num, X_test_num, X_train_img, X_test_img, y_train, y_test = train_test_split(
        X_num, X_img, y, test_size=0.2, shuffle=False
    )
    
    # 4) scale numeric data
    scaler = StandardScaler()
    X_train_num_scl = scaler.fit_transform(X_train_num)
    X_test_num_scl  = scaler.transform(X_test_num)
    
    # 5) train numeric model (example: random forest)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_num_scl, y_train)
    
    # 6) train CNN model
    cnn = build_cnn_model(input_shape=(64,64,3))
    cnn.fit(
        X_train_img, y_train,
        validation_split=0.2,
        epochs=5,               # keep small for demonstration
        batch_size=32,
        verbose=1
    )
    
    # 7) get probabilities from each model
    rf_probs_test = rf.predict_proba(X_test_num_scl)[:,1]  # shape (# test)
   
