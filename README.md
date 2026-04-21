# Titanic Survival Predictor

A web-based machine learning application that predicts passenger survival probability on the Titanic using a Random Forest classifier trained on the Kaggle Titanic dataset.

## Features

- **Interactive Web Interface**: Beautiful, responsive HTML5 UI with ocean-themed dark design
- **Real-time Predictions**: Get instant survival probability predictions
- **Random Forest Model**: Trained on historical Titanic passenger data
- **Prediction History**: Track all predictions made during the session
- **Factor Analysis**: Visual indicators showing which factors positively/negatively influence survival
- **REST API**: Flask backend for easy integration

## Tech Stack

- **Backend**: Flask, Flask-CORS
- **ML**: scikit-learn (Random Forest Classifier), pandas, numpy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Data Processing**: pandas, seaborn, matplotlib

## Project Structure

```
Titanic/
├── app.py                          # Flask REST API
├── train_model.py                  # Model training script
├── train.csv                        # Titanic dataset (for training)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── titanic_predictor.html          # Standalone HTML version
├── model/
│   ├── rf_model.pkl               # Trained Random Forest model
│   ├── scaler.pkl                 # StandardScaler for feature normalization
│   └── encoder_info.json          # Category encodings (sex, embarked)
└── templates/
    └── titanic_predictor.html      # Flask-served HTML template
```

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/titanic-predictor.git
   cd Titanic
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   - Navigate to: `http://127.0.0.1:5000`
   - You should see the Titanic Survival Predictor interface

3. **Make predictions**
   - Fill in passenger details (class, gender, age, fare, etc.)
   - Click "⚓ Predict Survival Chance"
   - View the survival probability and contributing factors

### Training the Model (Optional)

If you want to retrain the model from scratch:

```bash
python train_model.py
```

This will:
- Load the Titanic dataset from `train.csv`
- Perform feature engineering
- Train a Random Forest classifier
- Save model artifacts to `model/` directory

## API Endpoints

### GET `/`
Serves the HTML predictor interface

### POST `/predict`

**Request body (JSON):**
```json
{
  "pclass": 3,
  "sex": "male",
  "age": 28,
  "sibsp": 0,
  "parch": 0,
  "fare": 14.5,
  "embarked": "S"
}
```

**Response (JSON):**
```json
{
  "survived": false,
  "probability": 0.18,
  "percent": 18
}
```

**Parameters:**
- `pclass` (int, 1-3): Passenger class
- `sex` (string): "male" or "female"
- `age` (float): Age in years
- `sibsp` (int): Number of siblings/spouses aboard
- `parch` (int): Number of parents/children aboard
- `fare` (float): Ticket fare in pounds
- `embarked` (string): Port of embarkation - "S" (Southampton), "C" (Cherbourg), "Q" (Queenstown)

## Model Details

- **Algorithm**: Random Forest Classifier (20 trees)
- **Training Data**: Kaggle Titanic dataset (~891 passengers)
- **Features Used**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone
- **Target**: Survival (binary: 0 = perished, 1 = survived)

### Key Survival Factors
- **Gender**: Females had significantly higher survival rates ("women and children first" policy)
- **Passenger Class**: 1st class passengers had better access to lifeboats
- **Age**: Children had higher survival rates
- **Fare**: Higher fares indicate better accommodations and location on ship
- **Family Size**: Traveling with family (but not too large) improved chances

## Development

### Adding Features
To add new prediction features:

1. Update the input form in `templates/titanic_predictor.html`
2. Add the feature to the `FEATURES` list in `app.py`
3. Retrain the model with new data

### Modifying Styling
The CSS is embedded in `templates/titanic_predictor.html`. Key color variables:
- `--navy`: Main background color
- `--ice`: Accent/highlight color
- `--gold`: Accent lines

## Troubleshooting

**White screen on localhost:5000?**
- Ensure Flask is running: check terminal for "Running on http://127.0.0.1:5000"
- Clear browser cache (Ctrl+Shift+Delete)
- Check browser console for JavaScript errors (F12)

**Model not found error?**
- Run `python train_model.py` to generate model files

**CORS errors?**
- Flask-CORS is already configured. Check that requests are coming from `http://127.0.0.1:5000`

## Future Enhancements

- [ ] Add more features (cabin, ticket, name)
- [ ] Implement input validation/error handling
- [ ] Add confidence intervals to predictions
- [ ] Model comparison (XGBoost, Neural Networks)
- [ ] Mobile-responsive improvements
- [ ] Batch prediction upload (CSV)
- [ ] Model explainability (SHAP values)

## License

This project is for educational purposes. Dataset sourced from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).

## Author

Created as a mini project for AISSMS IOIT, Department of Computer Engineering (2025-26)

## Contact

For issues or questions, please open an issue on GitHub.
