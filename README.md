# Building-OCR

This project uses a Convolutional Neural Network (CNN) to classify handwritten **math symbols** and **digits (0–9)** from grayscale images of size `100x100`.

### 📊 Dataset Overview

* **Total images**: 10,001
* **Image size**: 100 x 100 pixels (grayscale)
* **Classes**: Digits (0–9), Math operators (`add`, `sub`, `mul`, `div`, `eq`, `dec`), Variables (`x`, `y`, `z`)
* **Max class**: `sub` → 655 images
* **Min class**: `z` → 212 images
* **Imbalance Ratio**: ≈ 3.09 (most common class has 3x more data than the least)

![download](https://github.com/user-attachments/assets/f57b9400-3c90-4c11-8aab-495371fc04ac)

---

### 🔢 **Raw Data Point (From CSV)**

The **input to the model** comes from a CSV file where each row represents a **100×100 grayscale image** of a math symbol that has been **flattened** into a single row of 10,000 pixel values.<br>
<br>
Each row in the CSV looks like this:

```
label, pixel_0, pixel_1, ..., pixel_9999
3,     0,       0,       ..., 255
```

* `label`: The actual symbol class (e.g., `'3'`, `'+'`, `'x'`).
* `pixel_0` to `pixel_9999`: Each is a grayscale value from 0 (black) to 255 (white).

---

 ### 🧼 **Preprocessing Steps**

You apply these transformations before feeding into the CNN:

1. **Label encoding**: Strings like `'3'` or `'add'` are converted to integers.
2. **Normalization**: All pixel values are divided by 255.0 so they fall between `0` and `1`.
3. **Reshape**: Each flattened image of shape `(10000,)` is reshaped to `(100, 100, 1)` to match CNN input:

   * `100 × 100` is the image size
   * `1` is the number of channels (grayscale)

**single preprocessed image**:

```python
plt.imshow(features[0].reshape(100, 100), cmap='gray')
plt.title("Example Input Image")
plt.show()
```
This will show a black-and-white image of a digit/symbol—exactly how the model "sees" it.

The model gets input in the shape:

```python
(batch_size, 100, 100, 1)
```

---
### Model Architecture

We use a **Sequential CNN model** with multiple convolutional and pooling layers, followed by dense layers to classify the symbols.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

#### Layer-by-Layer Explanation:

| Layer                   | Description                                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Conv2D (32 filters)** | Extracts **local spatial features** using 3x3 kernels. It detects edges, curves, and textures in small areas of the image. |
| **MaxPooling2D (2x2)**  | Reduces dimensionality by keeping only the **strongest feature activations** (like zooming out). Helps avoid overfitting.  |
| **Conv2D (64 filters)** | Captures **more complex patterns** (e.g., parts of symbols, loops, intersections) with deeper filters.                     |
| **MaxPooling2D**        | Again, shrinks feature maps to focus on the most important features.                                                       |
| **Flatten**             | Converts 2D feature maps into a **1D vector** for input into dense layers.                                                 |
| **Dense (128 units)**   | Fully connected layer that learns **abstract combinations** of the extracted features.                                     |
| **Dense (Output)**      | Uses **softmax** to output class probabilities (one for each symbol).                                                      |

---
### ⚙️ Training Details

* **Optimizer**: Adam
* **Loss**: Sparse Categorical Crossentropy
* **Epochs**: 30
* **Batch size**: 32
* **Train/Validation Split**: 80/20

---
### Classification Report
You'll see something like this:
| Term          | Simple Meaning                                         |
| ------------- | ------------------------------------------------------ |
| **Precision** | Of all predicted as "this label", how many were right? |
| **Recall**    | Of all actual "this label", how many did we catch?     |
| **F1-score**  | A balance of precision & recall (ideal = 1.0)          |
| **Support**   | Number of test images for that label                   |


Example:<br>
If Label: 3 has low *precision*, it means the model often thinks something is a 3 when it’s not.<br>
If Label: 8 has low *recall*, the model misses a lot of actual 8s (e.g., it confuses them with 3s).<br>

---
### Confusion Matrix
It’s a big table like this:
| True\Pred | +   | -  | 3  | 8  |
| --------- | --- | -- | -- | -- |
| **+**     | 110 | 5  | 2  | 3  |
| **-**     | 3   | 88 | 6  | 3  |
| **3**     | 1   | 4  | 54 | 31 |
| **8**     | 2   | 1  | 22 | 70 |

What to Look For:
 - Diagonal values (↘) are correct predictions. Higher = better.
 - Off-diagonal values are misclassifications.
<br>
Example Analysis: <br>
If 31 “3”s were predicted as “8” → that’s a big confusion. <br>

---
### 📈 Results

* Final Train Accuracy: 95.93%
* Final Validation Accuracy: 98.16%

![download](https://github.com/user-attachments/assets/a0944345-8c14-448d-8a6e-2c72b6f5d442)

---

### 🚧 Limitations & Future Improvements

#### 🔻 Current Limitations

* A custom **UI was implemented** for drawing digits or math symbols, but the model trained from scratch (`best_model.h5`) often produces **incorrect predictions** when integrated with this interface. This is likely due to:

  * Input variation between dataset images and drawn UI inputs.
  * Limited generalization capacity of the trained model.
* The model was trained on clean, preprocessed CSV images, which differ from real-world strokes drawn on a canvas.

#### 🚀 Suggested Improvements

* Replace or augment the current model with **pretrained OCR models** such as:

  * **TrOCR** (Transformer-based OCR by Microsoft) for character-level recognition.
  * **YOLOv5 or YOLOv8** for object detection-based segmentation of full handwritten equations.
* Use **data augmentation or domain adaptation** to bridge the gap between training images and UI-drawn inputs.
* Deploy the interface using platforms like **Streamlit Cloud** or **Hugging Face Spaces** for live testing with more flexible model backends.

---

