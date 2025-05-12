# Flight feel analyzer - Classification model

<div align="center">
  <img src="https://res.cloudinary.com/artevivo/image/upload/v1746795604/Presentaci%C3%B3n_Planificaci%C3%B3n_Viaje_Elegante_Fotogr%C3%A1fico_Blanco_2_q55bvz.jpg" alt="Banner centrado" width="900" height="450">
</div>

## 📌 Index
- [About the Project](#-about-the-project)  
- [Main Features](#-main-features)  
- [Current Issues](#-current-issues)
- [Folder Structure](#-folder-structure)
- [Possible Improvements](#-possible-improvements)  
- [EDA Architecture Diagram](#-eda-architecture-diagram)  
- [Installation and Usage](#-installation-and-usage)
- [Model Performance & Hyperparameters](#-model-performance-&-hyperparameters)
- [Testing](#-testing)
- [Render Deployment](#-render-deployment)
- [Collaborators](#-collaborators)  
---

## ✈️ About the Project

**Flight Feel Analyzer** is a machine learning classification project developed after being contracted by an airline to help predict passenger satisfaction levels. 

The goal is to use customer data and their flight survey responses to determine whether they were **satisfied or not** with the service. After a rigorous process of **exploratory data analysis (EDA)** and data preprocessing ([Colab](https://colab.research.google.com/drive/1cKj-GHQK5rYj1P3glOpQzHk1-dhlbI76?usp=sharing)
), we selected a **Random Forest Classifier** as our final model due to its robustness and accuracy for binary classification tasks.

The model is integrated into a **Flask** backend application, where users input their flight details, and the system returns a satisfaction prediction. The data is managed using **MySQL Workbench** with **Flask-SQLAlchemy**, and development has been carried out in **Python**, using **Jupyter Notebooks** and **VSCode**.

---
## 🔍 Main Features  
✅ Complete EDA process with data cleaning and visualizations.  
✅ Binary classification with **Random Forest** model.  
✅ Backend implemented with **Flask** and **Flask-SQLAlchemy**.  
✅ Database integration using **MySQL Workbench**.  
✅ Well-organized and modular project structure.  

---

## 🐞 Current Issues  
❌ The model's performance could benefit from using more historical data.

---

## 💡 Possible Improvements  
✅ Implement additional models for comparison (e.g., Logistic Regression, XGBoost).  
✅ Add frontend using Streamlit or React.  
✅ Improve model explainability with SHAP values.  

---

## 📁 Folder Structure

```bash
# Flight Feel Analyzer
📂 Flight-Feel-Analyzer/
├── 📂 .venv/
├── 📂 app/
│   └── 📂 ml_models
│   └── 📂 models
│   └── 📂 static
│   └── 📂 templates
│   └── _init_.py
│   └── model_loader.py
│   └── routes.py               
├── 📂 classification-model/  
│   └── model.pkl
│   └── testing-model.ipynb  
├── 📂 data/
├── 📂 EDA/
│   └── satisfaction-passenger.ipynb
├── 📂 tests/
│   └── test_model_satisfaction.py
├── 📜 .env 
├── 📜 README.md  
├── 📜 .gitignore  
├── 📜 requirements.txt
├── 📜 run.py  
```
---

## 🧠 EDA Architecture Diagram
<div align="center">
  <img src="https://res.cloudinary.com/artevivo/image/upload/v1746779498/Captura_de_pantalla_2025-05-09_092602_vrcdea.png" alt="Banner centrado" width="900" height="400">
</div>
---

## ⚙️ Installation and Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Flight-Feel-Analyzer.git
cd Flight-Feel-Analyzer
```
### 2️⃣ Create and activate the virtual environment
```bash
python -m venv .venv
source .venv/Scripts/activate # Windows
source .venv/bin/activate  # Linux/Mac
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Start the Flask App
```bash
flask run
```
---
## 📊 Model Performance & Hyperparameters

<div align="center">
  <img src="https://res.cloudinary.com/artevivo/image/upload/v1746792966/Captura_de_pantalla_2025-05-08_082125_bh38io.png" alt="Metrics" width="800" height="250">
</div>
---

## 🧐 Testing
Copy the following command to run the tests:
```bash
python -m unittest tests/test_model_satisfaction.py
```
<div align="center">
  <img src="https://res.cloudinary.com/artevivo/image/upload/v1747053472/Captura_de_pantalla_2025-05-12_134230_r3w2qg.png" alt="Metrics" width="400" height="100">
</div>
---

## 🚀 Render Deployment
You can view the live version of the Flight Feel Analyzer project at [Render - Flight Feel Analyzer](https://flight-feel-analyzer-1.onrender.com).

---

## 🧑‍💻 Collaborators
This project was developed by the following contributors:
- [Andreina Suescum](https://github.com/mariasuescumg/mariasuescumg/)  
- [Polina ](https://github.com/fintihlupik/)   
- [Nhoeli Salazar](https://www.linkedin.com/in/nhoeli-salazar/)   
- [Omar Lengua Suárez](https://github.com/Omarlsant/)
---
<p align="right">(<a href="#-index">⬆️ Back to top</a>)</p>
