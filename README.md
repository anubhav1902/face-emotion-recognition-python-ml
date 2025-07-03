## 😃 Face Emotion Recognition using Python & Machine Learning

This project is all about teaching a computer to understand how you're feeling — just by looking at your face through a webcam! It can tell if you're happy, sad, angry, surprised, scared, disgusted, or neutral.

We trained a model using a dataset of facial expressions (FER-2013) and used Python tools to make it work in real-time. Think of it like giving your computer a basic sense of emotional awareness.

---

## 🚀 What This Project Can Do

* 🧠 Understand your emotions using facial expressions
* 📸 Detect your face live using your webcam
* ⚡ Work in real-time, so no waiting
* 🛠️ Easy to set up with just a few Python libraries

---

## 🧰 What We Used

* **Python** – The main programming language
* **OpenCV** – To detect and capture your face using the camera
* **TensorFlow & Keras** – To build and train the emotion detection model
* **NumPy & Pandas** – For handling image and dataset information
* **Matplotlib** – For visualizing training results

---

## 📁 How the Project is Organized

```
face-emotion-recognition-python-ml/
│
├── dataset/                # Facial expression images for training
├── haarcascade/            # Face detection files
├── models/                 # Saved model after training
├── screenshots/            # Demo screenshots
├── src/                    # Main code files
│   ├── train_model.py      # Script to train the model
│   ├── predict_emotion.py  # Predict emotion from an image
│   └── webcam_demo.py      # Real-time demo with webcam
├── .gitignore              # Files to ignore when pushing to GitHub
├── requirements.txt        # Required Python libraries
└── README.md               # This file!
```

---

## 🧪 How to Run This Yourself

1. **Download the project**:

   ```bash
   git clone https://github.com/anubhav1902/face-emotion-recognition-python-ml.git
   cd face-emotion-recognition-python-ml
   ```

2. **Install the necessary Python packages**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Make sure the dataset is downloaded** and placed in the `dataset/` folder.

4. **Try it out using your webcam**:

   ```bash
   python src/webcam_demo.py
   ```

---

## 💻 Other Ways to Use It

* **Train the model yourself**:

  ```bash
  python src/train_model.py --epochs 30 --batch_size 64
  ```

* **Test it on an image instead of live camera**:

  ```bash
  python src/predict_emotion.py --image_path path/to/image.jpg
  ```

* **Run webcam demo anytime**:

  ```bash
  python src/webcam_demo.py
  ```

---

## 🤝 Want to Help?

Feel free to suggest improvements or contribute to the code. Just open an issue or submit a pull request.

---

## 📄 License

This project is open-source and uses the MIT License.

---

