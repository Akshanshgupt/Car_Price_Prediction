# 🚗 Car Price Prediction App

A **web application** built with **Streamlit** and **Lasso Regression** to predict the **resale price of cars** based on user inputs or bulk CSV data.

---

## 🔍 Features

- Predict the estimated car resale price from:
  - Manufacturing year
  - Kilometers driven
  - Fuel type (Petrol, Diesel, CNG)
  - Seller type (Individual, Dealer)
  - Transmission (Manual, Automatic)
- Display **price range** (Low, Estimated, High) with a chart.
- Show **feature importance** using Lasso coefficients.
- Calculate **model accuracy (R² score)** if actual prices are provided in a CSV.
- Upload a CSV for **bulk predictions** and download results.

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Streamlit** for the web interface
- **scikit-learn** for Lasso Regression
- **NumPy** & **Pandas** for data handling
- **Matplotlib** for visualization
- **Pickle** for saving/loading the trained model

---
📂 File Structure
├── app.py                  
├── car_price_model.pkl     
├── requirements.txt    
├── README.md               
└── sample_data.csv         
📝 Usage
1. Single Prediction

Enter car details in the app:

Manufacturing Year

Kilometers Driven

Fuel Type

Seller Type

Transmission

Click Predict Car Price.

View the estimated price and price range chart.

2. R² Score Calculation

Upload a CSV containing all features plus Actual_Price column.

The app calculates and displays the model's R² score.

3. Bulk Prediction

Upload a CSV with all features (columns must match training features order).

The app outputs predicted prices and allows you to download results as a CSV.

🚀 License

This project is open-source. Feel free to modify and use it for learning purposes.

Made with ❤️ by Akshansh Gupta

   


