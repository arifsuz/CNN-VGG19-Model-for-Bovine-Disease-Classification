# **Project Documentation: CNN Model Using VGG19 for Cattle Classification**

## **1. Dataset Preparation**
The first step involves organizing and preparing the dataset for training, validation, and testing.

### **Steps:**
1. **Dataset Extraction**  
   Extract the dataset from a compressed file (e.g., `.zip`) into a working directory.  
   ```python
   import zipfile
   with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
       zip_ref.extractall('dataset')
   ```

2. **Folder Structure**  
   Organize the dataset into `train`, `val`, and `test` directories, with subfolders for each class (e.g., `healthy`, `infected`).  
   ```python
   os.makedirs('dataset/train/healthy', exist_ok=True)
   os.makedirs('dataset/train/infected', exist_ok=True)
   ```

3. **Label Mapping**  
   Ensure that the folder names correspond to the class labels for proper loading during training.

---

## **2. Preprocessing**
This step ensures that the dataset is in the correct format and size for the model.

### **Steps:**
1. **Convert to Grayscale**  
   Convert images to grayscale to reduce complexity.  
   ```python
   gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
   cv2.imwrite(os.path.join(dst, img), gray)
   ```

2. **Recolor to RGB**  
   Convert grayscale images back to RGB to match the input requirements of the VGG19 model.  
   ```python
   rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
   cv2.imwrite(os.path.join(dst, img), rgb)
   ```

3. **Resize Images**  
   Resize all images to `(224, 224)` to match the input size of the VGG19 model.  
   ```python
   resized = cv2.resize(img, (224, 224))
   cv2.imwrite(img_path, resized)
   ```

4. **Data Augmentation**  
   Apply data augmentation techniques to increase dataset diversity.  
   ```python
   augment_gen = ImageDataGenerator(
       rotation_range=20,
       width_shift_range=0.1,
       height_shift_range=0.1,
       zoom_range=0.1,
       horizontal_flip=True,
       fill_mode='nearest'
   )
   ```

---

## **3. Model Development**
The model is built using the VGG19 architecture pre-trained on ImageNet.

### **Steps:**
1. **Load Pre-trained Model**  
   Load the VGG19 model without the top layers and freeze the initial layers.  
   ```python
   base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
   for layer in base_model.layers[:15]:
       layer.trainable = False
   ```

2. **Add Custom Layers**  
   Add custom layers for classification based on the number of classes.  
   ```python
   x = GlobalAveragePooling2D()(base_model.output)
   x = Dropout(0.3)(x)
   x = Dense(128, activation='relu')(x)
   x = Dropout(0.3)(x)
   predictions = Dense(train_data.num_classes, activation='softmax')(x)
   model = Model(inputs=base_model.input, outputs=predictions)
   ```

3. **Compile the Model**  
   Compile the model with the Adam optimizer and categorical cross-entropy loss.  
   ```python
   model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
   ```

4. **Train the Model**  
   Train the model using the training and validation datasets.  
   ```python
   history = model.fit(
       train_data,
       validation_data=val_data,
       epochs=15
   )
   ```

5. **Save the Model**  
   Save the trained model for future use.  
   ```python
   model.save('model_vgg19_cattle.h5')
   ```

---

## **4. Evaluation**
Evaluate the model's performance using the test dataset.

### **Steps:**
1. **Make Predictions**  
   Use the model to predict the test dataset.  
   ```python
   y_pred_probs = model.predict(test_data)
   y_pred = np.argmax(y_pred_probs, axis=1)
   y_true = test_data.classes
   ```

2. **Confusion Matrix**  
   Generate a confusion matrix to visualize prediction accuracy.  
   ```python
   cm = confusion_matrix(y_true, y_pred)
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
   ```

3. **Classification Report**  
   Generate a classification report to evaluate precision, recall, and F1-score.  
   ```python
   print(classification_report(y_true, y_pred, target_names=labels))
   ```

4. **Plot Accuracy and Loss**  
   Visualize training and validation accuracy and loss over epochs.  
   ```python
   plt.plot(history.history['accuracy'], label='Train Accuracy')
   plt.plot(history.history['val_accuracy'], label='Val Accuracy')
   ```

---

## **5. Conclusion**
- The model achieved high accuracy with minimal overfitting.
- Performance on minority classes can be improved by adding more data or using class balancing techniques.

### **Recommendations:**
- Collect more data for underrepresented classes.
- Use techniques like oversampling or class weighting to handle class imbalance.
- Optimize the model further for deployment in real-time applications.#   C N N - V G G 1 9 - M o d e l - f o r - B o v i n e - D i s e a s e - C l a s s i f i c a t i o n  
 