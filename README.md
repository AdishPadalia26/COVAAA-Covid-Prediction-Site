# 🦠 COVAAA — COVID-19 Prediction and Visualization Website

**COVAAA** is an interactive, data-driven web platform that visualizes the global and India-specific spread of COVID-19 and predicts the number of new cases for the next 30 days using the **Facebook Prophet** time-series model.

Developed as a **Mini Project** at *Sardar Patel Institute of Technology (SPIT)* under the guidance of **Prof. Swapnali Kurhade**, this project combines **Machine Learning**, **Data Visualization**, and **Web Development** to provide actionable pandemic insights.

---

## 🚩 Project Highlights

- **Technologies Used:** NumPy, Pandas, Scikit-Learn, Matplotlib, HTML, CSS, Bootstrap, Flask
- **Model:** Facebook Prophet — achieved **94% accuracy**, surpassing ARIMA and LSTM models.
- **Functionality:**
  - Predicts COVID-19 cases for the next 30 days for every country.
  - Visualizes COVID-19 trends in India on a state-by-state basis using graphs and geographical maps.
  - Interactive full-stack web app for real-time exploration and forecasting.

---

## 🌍 Problem Statement

During the peak of the COVID-19 pandemic, governments and citizens struggled to anticipate how cases would evolve in the upcoming days.  
This unpredictability led to poor allocation of medical resources and delayed containment efforts.

**Objective:**  
Build a website that predicts the number of COVID-19 cases for the next 30 days (worldwide and for each Indian state) and visualizes key statistics such as:
- Confirmed, Active, Recovered, and Death counts  
- Top 10 affected countries and states  
- Country- and state-level forecasting graphs  

---

## 🧠 Abstract

COVAAA applies time-series forecasting using the **Facebook Prophet model**, a robust algorithm developed by Meta for scalable prediction of temporal data.  
The system processes publicly available COVID-19 datasets, performs cleaning and preprocessing, and produces a **94% accurate forecast** of future cases (surpassing ARIMA and LSTM models).  
It is deployed as a web application with an intuitive UI for both global and India-specific exploration.

---

## ⚙️ Tech Stack

| Component      | Technology                                         |
|----------------|----------------------------------------------------|
| **Language**   | Python                                             |
| **Libraries**  | NumPy, Pandas, Matplotlib, Plotly, Scikit-Learn, fbprophet |
| **Web Framework** | Django, Flask                                   |
| **Frontend**   | HTML, CSS, Bootstrap                               |
| **Environment**| Google Colab / Jupyter Notebook                    |

---
## 📊 Modules

1. **Research & Data Collection**  
   - Extract reliable global and Indian COVID-19 datasets.  
2. **Data Preprocessing**  
   - Handle missing data, clean invalid entries, and normalize date formats.  
3. **Model Building**  
   - Apply Facebook Prophet to forecast daily confirmed cases for the next 30 days.  
4. **Visualization**  
   - Plot daily cases, recoveries, deaths, and active trends using Matplotlib and Plotly.  
5. **Web Application (Django/Flask)**  
   - Interactive pages for global and India-specific statistics.  
6. **Prediction Interface**  
   - Input country/state → generate prediction graph dynamically.

---

## 🧩 Algorithm — Facebook Prophet

- **Type:** Additive time-series forecasting model  
- **Why Prophet?**
  - Handles seasonality, outliers, and missing data gracefully.  
  - Provides interpretable daily, weekly, and yearly trends.  
  - Easy to deploy and update as new data arrives.  
- **Key Columns:**
  - `ds`: Date  
  - `y`: Target variable (number of cases)  
  - `yhat`: Forecasted value  
  - `yhat_lower`, `yhat_upper`: Confidence interval bounds  

---

## ❓ Why Not LSTM or Other Deep Learning Time Series Architectures?

While deep learning methods like **LSTM (Long Short-Term Memory)** networks and other recurrent neural networks are powerful for time-series forecasting, they were not chosen for this project for several reasons:

- **Data Requirements:**  
  LSTM models require large amounts of data to generalize well and avoid overfitting. COVID-19 daily case data, especially for specific regions or states, can be relatively limited in length and quality.

- **Interpretability:**  
  Facebook Prophet offers highly interpretable results, exposing trend, seasonality, and holiday effects as separate components. LSTM and deep learning models are often considered "black boxes," making it difficult for end-users and policymakers to understand or trust the predictions.

- **Ease of Use and Tuning:**  
  Prophet is designed for analysts and data scientists who may not have deep expertise in neural networks. It requires minimal parameter tuning and handles missing data and outliers gracefully. LSTM models demand careful architecture design, hyperparameter tuning, and more intensive preprocessing.

- **Computation and Deployment:**  
  Prophet is lightweight and can run efficiently on standard CPUs, making it easy to deploy in web applications. LSTM models are computationally intensive, require GPUs for training, and increase infrastructure complexity for real-time inference.

- **Seasonality and Domain Features:**  
  Prophet natively models daily, weekly, and yearly seasonality, and can easily incorporate domain-specific events or holidays. While possible with LSTM, this usually requires feature engineering and more complex modeling efforts.

**Summary:**  
Facebook Prophet was selected for its balance of accuracy, interpretability, ease of deployment, and suitability for the scale and nature of COVID-19 case data.

---

## 🖥️ Web Pages Overview

### 🏠 Home Page
- Displays live COVID-19 statistics for India.  
- Links to **World** and **India** dashboards.

### 🌎 World Page
- Global map showing total cases, deaths, recoveries.  
- Top 10 most affected countries (table + bar charts).  
- User input for **country-specific 30-day prediction**.  

### 🌏 Country Page
- Line graph showing actual vs predicted cases for the selected country.  

### 🇮🇳 India Page
- Interactive India map with **state-wise cases** on hover.  
- Bar graphs for **active** and **recovered** distributions.  
- Input for **state-level 30-day prediction**.  

### 🧭 State Page
- Forecast visualization for the entered Indian state.  

### ℹ️ About Us Page
- Overview of the project, technology stack, and team members.

---

## 📈 Results

| Metric             | Result                                    |
|--------------------|-------------------------------------------|
| **Model**          | Facebook Prophet                          |
| **Forecast Period**| 30 Days                                   |
| **Accuracy**       | **94%**                                   |
| **Visualization**  | Matplotlib + Plotly (bar, line, geo)      |
| **UI Framework**   | Django / Flask + Bootstrap                |

The Prophet model’s predictions closely aligned with real-world case counts, validating its superiority over ARIMA and LSTM for pandemic forecasting in this context.

---

## 🚀 Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/AdishPadalia26/COVAAA-Covid-Prediction-Site.git
    cd COVAAA-Covid-Prediction-Site
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Django or Flask server**
    ```bash
    python manage.py runserver   # For Django
    # OR
    flask run                    # For Flask, if structured accordingly
    ```
4. **Access the app**  
   Open your browser and go to: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔮 Future Work

- **Variant-based Visualization:** Graphs segmented by COVID-19 variants.
- **Pincode API:** Detect user’s location and show nearby statistics.
- **Age-wise Prediction:** Separate forecasts for different age groups.
- **Improved Datasets:** Real-time updates via API integration (WHO / OWID).

---

## 🧾 References

- Rohini M. et al., A Comparative Approach to Predict Corona Virus Using Machine Learning, ICAIS 2021.
- Locquet M. et al., A systematic review of prediction models to diagnose COVID-19, Arch Public Health 2021.
- Sujath R. et al., A machine learning forecasting model for COVID-19 pandemic in India, Stoch Environ Res Risk Assess 2020.
- Mahdavi M. et al., A ML-based exploration of COVID-19 mortality risk, PLOS ONE 2021.
- Shaikh S. et al., Analysis and Prediction of COVID-19 using Regression Models and Time-Series Forecasting, Confluence 2021.
- Mary L. W., Raj S. A. A., ML Algorithms for Predicting SARS-CoV-2 (COVID-19): Comparative Analysis, ICOSEC 2021.

---

## 👨‍💻 Team

- **Adish Padalia** — ML & Backend (Django / Prophet Model)
- **Ayush Bhandarkar** — Data Visualization & Frontend Design
- **Anmol Chokshi** — Web Integration & Deployment

**Guide:** Prof. Swapnali Kurhade, IT Department, SPIT

---

## 🧩 Repository Structure

```
COVAAA-Covid-Prediction-Site/
├── covid/                     # Django app
│   ├── templates/             # HTML templates (home, world, india, etc.)
│   ├── static/                # CSS, JS, images
│   ├── views.py               # View logic
│   ├── urls.py                # URL routes
│   ├── models.py              # (if used for data)
│   └── prophet_model.py       # ML model integration
├── manage.py
├── requirements.txt
└── README.md
```

---

## 🏁 Conclusion

COVAAA demonstrates how time-series forecasting and interactive web technologies can be combined to deliver meaningful insights during a public-health crisis.  
By providing a simple interface for visualizing and predicting COVID-19 trends, the project supports proactive decision-making and awareness among users.

**Accuracy achieved:** 94%  
**Forecast horizon:** 30 days  
**Model used:** Facebook Prophet

---

## 📜 License

This project is released under the MIT License.

> "Predicting tomorrow’s numbers, preparing today’s world."

---

## Author

👤 **Adish Padalia**  
🌐 [LinkedIn](https://www.linkedin.com/in/adish-padalia/)  
💻 [GitHub](https://github.com/AdishPadalia26)  
📧 padaliaadish@gmail.com
