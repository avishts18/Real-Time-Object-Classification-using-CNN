> Real-Time Object Classification using CNN

> Setup Instructions

 1. Clone the Repository

```bash
git clone https://github.com/avishts18/Real-Time-object-Classificatiion-using-CNN.git
cd Real-Time-object-Classificatiion-using-CNN
```
# The full project structure is not available in the above GitHub link.

 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate     # Windows
```

 3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Train the Model

```bash
python model.py
```

This trains the CNN on MNIST or your dataset and saves `trained model`.

5. Run Real-Time Inference

```bash
python main.py
```

This opens your webcam and displays live predictions.

---

> Optional: Web Interface (Flask)

Install Flask if not already done:

```bash
pip install flask
```

Run:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---



> requirements.txt 

```
tensorflow
opencv-python
flask
matplotlib
numpy
streamlit
```
> Screenshots
![Result-1](https://github.com/user-attachments/assets/e9d7141d-fb2b-44cf-a21d-49091c259a84)
![Result-2](https://github.com/user-attachments/assets/0b94be34-bc7d-4b7f-85f2-a8d0c83fb869)
