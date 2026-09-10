# AI Nirmaan

A machine learning web application built with [Streamlit](https://streamlit.io/) that provides interface access to trained classification and regression models.

## 🚀 Live Demo

Access the deployed application here: [AI Nirmaan App](https://ainirmaan-f5lrru6bfxyuk8uadipeah.streamlit.app/)

---

## 📌 Features

* **Classification Predictions:** Interactive predictions using pre-trained and scaled classification algorithms (`best_classification_model.pkl`, `classification_scaler.pkl`).
* **Regression Predictions:** Continuous value predictions using trained regression models (`best_regression_model.pkl`).
* **Modular Codebase:** Organized structure separating UI layout (`app.py`) and model logic (`models.py`).

---

## 🛠️ Project Structure

```text
Ai_nirmaan/
│
├── app.py                         # Main Streamlit application entry point
├── models.py                      # Model loading and inference helper functions
├── best_classification_model.pkl  # Trained classification model
├── best_regression_model.pkl      # Trained regression model
├── classification_scaler.pkl     # Feature scaler for classification data
├── requirements.txt               # Python dependencies
├── LICENSE                        # License file
└── README.md                      # Project documentation

💻 Local Setup Instructions
Prerequisites
Ensure you have Python 3.8+ installed on your machine.

Installation
Clone the repository:

Bash
git clone [https://github.com/sahildethe06-png/Ai_nirmaan.git](https://github.com/sahildethe06-png/Ai_nirmaan.git)
cd Ai_nirmaan
Create and activate a virtual environment (optional but recommended):

Windows:

Bash
python -m venv venv
.\venv\Scripts\activate
macOS/Linux:

Bash
python3 -m venv venv
source venv/bin/activate
Install the dependencies:

Bash
pip install -r requirements.txt
Run the Streamlit application:

Bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501.

📜 License
This project is open source and available under the MIT License.
