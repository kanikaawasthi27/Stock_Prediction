import os
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Stock Prediction System",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "models"
)

SCALERS_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "scalers"
)

RESULTS_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "results"
)

os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(SCALERS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

TARGET_COLUMN = "Close"

SEQUENCE_LENGTH = 60

LSTM_UNITS = 64
DROPOUT_RATE = 0.20
DENSE_UNITS = 32

EPOCHS = 30
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5


# ============================================================
# COMPANY CATALOGUE
# ============================================================

COMPANIES = [

    # USA
    {"name": "Apple Inc.", "ticker": "AAPL", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Microsoft Corporation", "ticker": "MSFT", "country": "USA", "exchange": "NASDAQ"},
    {"name": "NVIDIA Corporation", "ticker": "NVDA", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Amazon.com Inc.", "ticker": "AMZN", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Tesla Inc.", "ticker": "TSLA", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Alphabet Inc.", "ticker": "GOOGL", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Meta Platforms Inc.", "ticker": "META", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Netflix Inc.", "ticker": "NFLX", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Adobe Inc.", "ticker": "ADBE", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Intel Corporation", "ticker": "INTC", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Advanced Micro Devices", "ticker": "AMD", "country": "USA", "exchange": "NASDAQ"},
    {"name": "PayPal Holdings", "ticker": "PYPL", "country": "USA", "exchange": "NASDAQ"},
    {"name": "Coca-Cola", "ticker": "KO", "country": "USA", "exchange": "NYSE"},
    {"name": "Walmart", "ticker": "WMT", "country": "USA", "exchange": "NYSE"},
    {"name": "McDonald's", "ticker": "MCD", "country": "USA", "exchange": "NYSE"},
    {"name": "Nike", "ticker": "NKE", "country": "USA", "exchange": "NYSE"},
    {"name": "JPMorgan Chase", "ticker": "JPM", "country": "USA", "exchange": "NYSE"},
    {"name": "Bank of America", "ticker": "BAC", "country": "USA", "exchange": "NYSE"},
    {"name": "Visa", "ticker": "V", "country": "USA", "exchange": "NYSE"},
    {"name": "Mastercard", "ticker": "MA", "country": "USA", "exchange": "NYSE"},

    # INDIA
    {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "country": "India", "exchange": "NSE"},
    {"name": "Tata Consultancy Services", "ticker": "TCS.NS", "country": "India", "exchange": "NSE"},
    {"name": "Infosys", "ticker": "INFY.NS", "country": "India", "exchange": "NSE"},
    {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "country": "India", "exchange": "NSE"},
    {"name": "ICICI Bank", "ticker": "ICICIBANK.NS", "country": "India", "exchange": "NSE"},
    {"name": "State Bank of India", "ticker": "SBIN.NS", "country": "India", "exchange": "NSE"},
    {"name": "Larsen & Toubro", "ticker": "LT.NS", "country": "India", "exchange": "NSE"},
    {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS", "country": "India", "exchange": "NSE"},
    {"name": "ITC", "ticker": "ITC.NS", "country": "India", "exchange": "NSE"},
    {"name": "Hindustan Unilever", "ticker": "HINDUNILVR.NS", "country": "India", "exchange": "NSE"},
    {"name": "Tata Motors", "ticker": "TATAMOTORS.NS", "country": "India", "exchange": "NSE"},
    {"name": "Mahindra & Mahindra", "ticker": "M&M.NS", "country": "India", "exchange": "NSE"},
    {"name": "Maruti Suzuki", "ticker": "MARUTI.NS", "country": "India", "exchange": "NSE"},
    {"name": "Asian Paints", "ticker": "ASIANPAINT.NS", "country": "India", "exchange": "NSE"},
    {"name": "Titan Company", "ticker": "TITAN.NS", "country": "India", "exchange": "NSE"},
    {"name": "Bajaj Finance", "ticker": "BAJFINANCE.NS", "country": "India", "exchange": "NSE"},
    {"name": "HCL Technologies", "ticker": "HCLTECH.NS", "country": "India", "exchange": "NSE"},
    {"name": "Wipro", "ticker": "WIPRO.NS", "country": "India", "exchange": "NSE"},
    {"name": "Tech Mahindra", "ticker": "TECHM.NS", "country": "India", "exchange": "NSE"},
    {"name": "Sun Pharmaceutical", "ticker": "SUNPHARMA.NS", "country": "India", "exchange": "NSE"},

    # OTHER COUNTRIES
    {"name": "Toyota Motor", "ticker": "TM", "country": "Japan", "exchange": "NYSE"},
    {"name": "Sony Group", "ticker": "SONY", "country": "Japan", "exchange": "NYSE"},
    {"name": "Alibaba", "ticker": "BABA", "country": "China", "exchange": "NYSE"},
    {"name": "JD.com", "ticker": "JD", "country": "China", "exchange": "NASDAQ"},
    {"name": "Samsung Electronics", "ticker": "005930.KS", "country": "South Korea", "exchange": "KOSPI"}
]

company_df = pd.DataFrame(COMPANIES)


# ============================================================
# TEMPORAL ATTENTION
# ============================================================

class TemporalAttention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        score = tf.nn.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context_vector = inputs * attention_weights

        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector


# ============================================================
# TICKER FILE NAME
# ============================================================

def ticker_to_filename(ticker):

    ticker = str(ticker).strip().upper()

    return (
        ticker
        .replace(".", "_")
        .replace("^", "_")
    )


# ============================================================
# MODEL PATHS
# ============================================================

def get_model_paths(ticker):

    safe_ticker = ticker_to_filename(ticker)

    return {

        "vanilla":
        os.path.join(
            MODELS_FOLDER,
            f"{safe_ticker}_vanilla_lstm.keras"
        ),

        "attention":
        os.path.join(
            MODELS_FOLDER,
            f"{safe_ticker}_lstm_attention.keras"
        ),

        "scaler":
        os.path.join(
            SCALERS_FOLDER,
            f"{safe_ticker}_scaler.joblib"
        ),

        "results":
        os.path.join(
            RESULTS_FOLDER,
            f"{safe_ticker}_dynamic_results.json"
        )
    }


# ============================================================
# CHECK WHETHER MODEL EXISTS
# ============================================================

def models_exist(ticker):

    paths = get_model_paths(ticker)

    return (
        os.path.exists(paths["vanilla"])
        and
        os.path.exists(paths["attention"])
        and
        os.path.exists(paths["scaler"])
    )


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(ttl=900)
def download_stock_data(
    ticker,
    period="5y"
):

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:

        raise ValueError(
            f"No data was found for ticker {ticker}."
        )

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = data[required].copy()

    data = data.dropna()

    data = data[
        data["Close"] > 0
    ]

    if len(data) < 200:

        raise ValueError(
            "Not enough historical data for training."
        )

    return data


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_time_series(data):

    total_rows = len(data)

    train_end = int(
        total_rows * 0.70
    )

    validation_end = int(
        total_rows * 0.85
    )

    train_data = data.iloc[
        :train_end
    ].copy()

    validation_data = data.iloc[
        train_end:validation_end
    ].copy()

    test_data = data.iloc[
        validation_end:
    ].copy()

    return (
        train_data,
        validation_data,
        test_data
    )


# ============================================================
# SCALE DATA
# ============================================================

def scale_data(
    train_data,
    validation_data,
    test_data
):

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    train_scaled = scaler.fit_transform(
        train_data[FEATURE_COLUMNS]
    )

    validation_scaled = scaler.transform(
        validation_data[FEATURE_COLUMNS]
    )

    test_scaled = scaler.transform(
        test_data[FEATURE_COLUMNS]
    )

    return (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler
    )


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    values,
    sequence_length=60
):

    X = []
    y = []

    close_index = FEATURE_COLUMNS.index(
        TARGET_COLUMN
    )

    for i in range(
        sequence_length,
        len(values)
    ):

        X.append(
            values[
                i-sequence_length:i
            ]
        )

        y.append(
            values[
                i,
                close_index
            ]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# CONTEXT SEQUENCES
# ============================================================

def create_context_sequences(
    previous_values,
    current_values,
    sequence_length=60
):

    combined = np.vstack(
        [
            previous_values,
            current_values
        ]
    )

    X = []
    y = []

    current_start = len(
        previous_values
    )

    close_index = FEATURE_COLUMNS.index(
        TARGET_COLUMN
    )

    for i in range(
        current_start,
        len(combined)
    ):

        if i < sequence_length:
            continue

        X.append(
            combined[
                i-sequence_length:i
            ]
        )

        y.append(
            combined[
                i,
                close_index
            ]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# COMPLETE PREPROCESSING
# ============================================================

def preprocess_data(data):

    (
        train_data,
        validation_data,
        test_data
    ) = split_time_series(data)

    (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler
    ) = scale_data(
        train_data,
        validation_data,
        test_data
    )

    X_train, y_train = create_sequences(
        train_scaled,
        SEQUENCE_LENGTH
    )

    X_validation, y_validation = (
        create_context_sequences(
            train_scaled[-SEQUENCE_LENGTH:],
            validation_scaled,
            SEQUENCE_LENGTH
        )
    )

    test_context = np.vstack(
        [
            train_scaled[-SEQUENCE_LENGTH:],
            validation_scaled
        ]
    )

    X_test, y_test = (
        create_context_sequences(
            test_context,
            test_scaled,
            SEQUENCE_LENGTH
        )
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        scaler
    )


# ============================================================
# VANILLA LSTM
# ============================================================

def build_vanilla_lstm():

    model = Sequential()

    model.add(
        LSTM(
            LSTM_UNITS,
            input_shape=(
                SEQUENCE_LENGTH,
                len(FEATURE_COLUMNS)
            )
        )
    )

    model.add(
        Dropout(
            DROPOUT_RATE
        )
    )

    model.add(
        Dense(
            DENSE_UNITS,
            activation="relu"
        )
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model


# ============================================================
# LSTM + ATTENTION
# ============================================================

def build_attention_model():

    inputs = Input(
        shape=(
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS)
        )
    )

    lstm_output = LSTM(
        LSTM_UNITS,
        return_sequences=True
    )(inputs)

    attention_output = TemporalAttention()(
        lstm_output
    )

    dropout_output = Dropout(
        DROPOUT_RATE
    )(attention_output)

    dense_output = Dense(
        DENSE_UNITS,
        activation="relu"
    )(dropout_output)

    output = Dense(1)(
        dense_output
    )

    model = Model(
        inputs=inputs,
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model


# ============================================================
# INVERSE TRANSFORM
# ============================================================

def inverse_transform_close(
    scaled_values,
    scaler
):

    scaled_values = np.asarray(
        scaled_values
    ).reshape(-1)

    dummy = np.zeros(
        (
            len(scaled_values),
            len(FEATURE_COLUMNS)
        )
    )

    close_index = FEATURE_COLUMNS.index(
        TARGET_COLUMN
    )

    dummy[:, close_index] = (
        scaled_values
    )

    original = scaler.inverse_transform(
        dummy
    )

    return original[
        :,
        close_index
    ]


# ============================================================
# MODEL TRAINING
# ============================================================

def train_models_for_ticker(
    ticker,
    period="5y",
    force_retrain=False
):

    ticker = str(
        ticker
    ).strip().upper()

    paths = get_model_paths(
        ticker
    )

    # --------------------------------------------------------
    # USE CACHE IF AVAILABLE
    # --------------------------------------------------------

    if (
        models_exist(ticker)
        and not force_retrain
    ):

        vanilla_model = (
            tf.keras.models.load_model(
                paths["vanilla"]
            )
        )

        attention_model = (
            tf.keras.models.load_model(
                paths["attention"],
                custom_objects={
                    "TemporalAttention":
                    TemporalAttention
                }
            )
        )

        scaler = joblib.load(
            paths["scaler"]
        )

        return (
            vanilla_model,
            attention_model,
            scaler,
            False,
            None
        )

    # --------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------

    st.write(
        "### 1️⃣ Downloading market data..."
    )

    data = download_stock_data(
        ticker,
        period
    )

    st.success(
        f"Downloaded {len(data):,} trading-day records."
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    st.write(
        "### 2️⃣ Preparing training data..."
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        scaler
    ) = preprocess_data(
        data
    )

    st.write(
        f"Training samples: {len(X_train):,}"
    )

    st.write(
        f"Validation samples: {len(X_validation):,}"
    )

    st.write(
        f"Testing samples: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # VANILLA LSTM
    # --------------------------------------------------------

    st.write(
        "### 3️⃣ Training Vanilla LSTM..."
    )

    vanilla_model = (
        build_vanilla_lstm()
    )

    vanilla_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    vanilla_history = (
        vanilla_model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                vanilla_stop
            ],
            shuffle=False,
            verbose=0
        )
    )

    st.success(
        "Vanilla LSTM training completed."
    )

    # --------------------------------------------------------
    # ATTENTION MODEL
    # --------------------------------------------------------

    st.write(
        "### 4️⃣ Training LSTM + Attention..."
    )

    attention_model = (
        build_attention_model()
    )

    attention_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    attention_history = (
        attention_model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                attention_stop
            ],
            shuffle=False,
            verbose=0
        )
    )

    st.success(
        "LSTM + Attention training completed."
    )

    # --------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------

    st.write(
        "### 5️⃣ Saving trained models..."
    )

    vanilla_model.save(
        paths["vanilla"]
    )

    attention_model.save(
        paths["attention"]
    )

    joblib.dump(
        scaler,
        paths["scaler"]
    )

    st.success(
        "Models and scaler saved successfully."
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    vanilla_scaled = (
        vanilla_model.predict(
            X_test,
            verbose=0
        )
    )

    attention_scaled = (
        attention_model.predict(
            X_test,
            verbose=0
        )
    )

    actual_prices = (
        inverse_transform_close(
            y_test,
            scaler
        )
    )

    vanilla_prices = (
        inverse_transform_close(
            vanilla_scaled,
            scaler
        )
    )

    attention_prices = (
        inverse_transform_close(
            attention_scaled,
            scaler
        )
    )

    vanilla_mae = mean_absolute_error(
        actual_prices,
        vanilla_prices
    )

    vanilla_rmse = np.sqrt(
        mean_squared_error(
            actual_prices,
            vanilla_prices
        )
    )

    attention_mae = mean_absolute_error(
        actual_prices,
        attention_prices
    )

    attention_rmse = np.sqrt(
        mean_squared_error(
            actual_prices,
            attention_prices
        )
    )

    return (
        vanilla_model,
        attention_model,
        scaler,
        True,
        {
            "vanilla_mae":
                float(vanilla_mae),

            "vanilla_rmse":
                float(vanilla_rmse),

            "attention_mae":
                float(attention_mae),

            "attention_rmse":
                float(attention_rmse)
        }
    )


# ============================================================
# CURRENT PREDICTION
# ============================================================

def generate_prediction(
    ticker,
    data,
    vanilla_model,
    attention_model,
    scaler
):

    latest_data = data[
        FEATURE_COLUMNS
    ].copy()

    latest_scaled = scaler.transform(
        latest_data
    )

    latest_sequence = (
        latest_scaled[
            -SEQUENCE_LENGTH:
        ]
    )

    latest_sequence = np.expand_dims(
        latest_sequence,
        axis=0
    )

    vanilla_scaled = (
        vanilla_model.predict(
            latest_sequence,
            verbose=0
        )
    )

    attention_scaled = (
        attention_model.predict(
            latest_sequence,
            verbose=0
        )
    )

    vanilla_prediction = (
        inverse_transform_close(
            vanilla_scaled,
            scaler
        )[0]
    )

    attention_prediction = (
        inverse_transform_close(
            attention_scaled,
            scaler
        )[0]
    )

    current_price = float(
        data["Close"].iloc[-1]
    )

    vanilla_change = (
        (
            vanilla_prediction
            - current_price
        )
        / current_price
    ) * 100

    attention_change = (
        (
            attention_prediction
            - current_price
        )
        / current_price
    ) * 100

    direction = (
        "UP"
        if attention_prediction > current_price
        else "DOWN"
    )

    return {

        "ticker": ticker,

        "last_date":
            str(
                data.index[-1].date()
            ),

        "current_price":
            current_price,

        "vanilla_prediction":
            float(
                vanilla_prediction
            ),

        "attention_prediction":
            float(
                attention_prediction
            ),

        "vanilla_change":
            float(
                vanilla_change
            ),

        "attention_change":
            float(
                attention_change
            ),

        "direction":
            direction
    }


# ============================================================
# SEARCH
# ============================================================

def search_companies(
    text
):

    text = str(
        text
    ).strip().lower()

    if not text:
        return company_df

    mask = (
        company_df["name"]
        .str.lower()
        .str.contains(
            text,
            na=False
        )
        |
        company_df["ticker"]
        .str.lower()
        .str.contains(
            text,
            na=False
        )
    )

    return company_df[
        mask
    ]


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 AI Stock Prediction System"
)

st.caption(
    "Dynamic stock analysis using Yahoo Finance, "
    "Vanilla LSTM and LSTM with Temporal Attention."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Prediction Settings"
    )

    period = st.selectbox(
        "Historical training period",
        [
            "1y",
            "2y",
            "5y",
            "10y"
        ],
        index=2
    )

    st.info(
        "The model uses the previous "
        "60 trading days to predict "
        "the next trading-day closing price."
    )

    st.divider()

    retrain = st.checkbox(
        "🔄 Retrain model",
        value=False
    )

    if retrain:

        st.warning(
            "Retraining may take several minutes."
        )


# ============================================================
# COMPANY SEARCH
# ============================================================

st.header(
    "🔎 Search for a Company"
)

search_text = st.text_input(
    "Search by company name or ticker",
    placeholder="Example: Microsoft, NVIDIA, Reliance, MSFT..."
)

search_results = search_companies(
    search_text
)

selected_ticker = None
selected_company = None


if not search_results.empty:

    options = [
        f"{row['name']} ({row['ticker']})"
        for _, row in search_results.iterrows()
    ]

    selected_option = st.selectbox(
        "Select company",
        options
    )

    selected_index = options.index(
        selected_option
    )

    selected_company = (
        search_results.iloc[
            selected_index
        ]
    )

    selected_ticker = (
        selected_company["ticker"]
    )


# ============================================================
# CUSTOM TICKER
# ============================================================

with st.expander(
    "⌨️ Or enter any Yahoo Finance ticker"
):

    custom_ticker = st.text_input(
        "Yahoo Finance ticker",
        placeholder="Example: MSFT, AAPL, RELIANCE.NS"
    )

    if custom_ticker.strip():

        selected_ticker = (
            custom_ticker
            .strip()
            .upper()
        )

        selected_company = {
            "name":
                selected_ticker,

            "ticker":
                selected_ticker,

            "country":
                "Yahoo Finance",

            "exchange":
                "Yahoo Finance"
        }


# ============================================================
# POPULAR COMPANIES
# ============================================================

st.header(
    "⭐ Popular Companies"
)

popular_tickers = [
    "AAPL",
    "NVDA",
    "MSFT",
    "TSLA",
    "RELIANCE.NS",
    "TCS.NS"
]

popular_cols = st.columns(
    len(popular_tickers)
)

for index, ticker in enumerate(
    popular_tickers
):

    row = company_df[
        company_df["ticker"] == ticker
    ]

    if row.empty:
        continue

    name = row.iloc[0]["name"]

    with popular_cols[index]:

        if st.button(
            name,
            key=f"popular_{ticker}",
            use_container_width=True
        ):

            st.session_state[
                "selected_ticker"
            ] = ticker

            st.rerun()


if "selected_ticker" in st.session_state:

    selected_ticker = (
        st.session_state[
            "selected_ticker"
        ]
    )

    matching = company_df[
        company_df["ticker"]
        == selected_ticker
    ]

    if not matching.empty:

        selected_company = (
            matching.iloc[0]
        )


# ============================================================
# COMPANY INFORMATION
# ============================================================

if selected_ticker:

    st.divider()

    if isinstance(
        selected_company,
        pd.Series
    ):

        company_name = (
            selected_company["name"]
        )

        country = (
            selected_company["country"]
        )

        exchange = (
            selected_company["exchange"]
        )

    else:

        company_name = selected_ticker
        country = "Yahoo Finance"
        exchange = "Yahoo Finance"

    st.header(
        f"📊 {company_name}"
    )

    st.write(
        f"**Ticker:** {selected_ticker}  "
        f"| **Exchange:** {exchange}  "
        f"| **Country:** {country}"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if selected_ticker:

    analyze = st.button(
        "🚀 Analyze & Predict",
        type="primary",
        use_container_width=True
    )

    if analyze:

        try:

            # ------------------------------------------------
            # TRAIN OR LOAD
            # ------------------------------------------------

            if models_exist(
                selected_ticker
            ) and not retrain:

                st.info(
                    "🧠 Existing trained model found. "
                    "Loading cached model..."
                )

            else:

                st.warning(
                    "🧠 No usable cached model found. "
                    "The system will train the model "
                    "for this company now."
                )

            with st.spinner(
                "Processing data and training/loading models..."
            ):

                data = download_stock_data(
                    selected_ticker,
                    period
                )

                (
                    vanilla_model,
                    attention_model,
                    scaler,
                    newly_trained,
                    metrics
                ) = train_models_for_ticker(
                    selected_ticker,
                    period,
                    retrain
                )

                prediction = generate_prediction(
                    selected_ticker,
                    data,
                    vanilla_model,
                    attention_model,
                    scaler
                )

            st.success(
                "✅ Prediction generated successfully."
            )

            if newly_trained:

                st.success(
                    "🎉 A new company-specific model "
                    "was trained and saved automatically."
                )

            else:

                st.info(
                    "⚡ Cached company-specific model "
                    "was used."
                )


            # =================================================
            # PREDICTION METRICS
            # =================================================

            st.subheader(
                "📌 Prediction Overview"
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Current Price",
                    f"{prediction['current_price']:.2f}"
                )

            with col2:

                st.metric(
                    "LSTM + Attention",
                    f"{prediction['attention_prediction']:.2f}",
                    f"{prediction['attention_change']:+.2f}%"
                )

            with col3:

                st.metric(
                    "Vanilla LSTM",
                    f"{prediction['vanilla_prediction']:.2f}",
                    f"{prediction['vanilla_change']:+.2f}%"
                )

            with col4:

                if prediction[
                    "direction"
                ] == "UP":

                    st.metric(
                        "Expected Direction",
                        "🟢 UP"
                    )

                else:

                    st.metric(
                        "Expected Direction",
                        "🔴 DOWN"
                    )


            # =================================================
            # PRICE CHART
            # =================================================

            st.subheader(
                "📈 Historical Closing Price"
            )

            chart_data = (
                data[
                    ["Close"]
                ]
                .tail(180)
            )

            st.line_chart(
                chart_data,
                use_container_width=True
            )


            # =================================================
            # MODEL COMPARISON
            # =================================================

            if metrics is not None:

                st.subheader(
                    "🤖 Model Performance"
                )

                metric_col1, metric_col2 = (
                    st.columns(2)
                )

                with metric_col1:

                    st.markdown(
                        "### Vanilla LSTM"
                    )

                    st.metric(
                        "MAE",
                        f"{metrics['vanilla_mae']:.4f}"
                    )

                    st.metric(
                        "RMSE",
                        f"{metrics['vanilla_rmse']:.4f}"
                    )

                with metric_col2:

                    st.markdown(
                        "### LSTM + Attention"
                    )

                    st.metric(
                        "MAE",
                        f"{metrics['attention_mae']:.4f}"
                    )

                    st.metric(
                        "RMSE",
                        f"{metrics['attention_rmse']:.4f}"
                    )

            else:

                st.info(
                    "Model was loaded from cache. "
                    "Training metrics are available "
                    "in the results generated during training."
                )


            # =================================================
            # RECENT DATA
            # =================================================

            st.subheader(
                "📋 Recent Market Data"
            )

            st.dataframe(
                data.tail(10),
                use_container_width=True
            )


            # =================================================
            # DATE
            # =================================================

            st.caption(
                "Last available trading date: "
                + prediction["last_date"]
            )


            # =================================================
            # DISCLAIMER
            # =================================================

            st.warning(
                "⚠️ This system is an educational/research "
                "machine-learning project. Predictions are "
                "not financial advice and actual market "
                "prices can differ significantly."
            )


        except Exception as error:

            st.error(
                "❌ Something went wrong."
            )

            st.exception(
                error
            )

else:

    st.info(
        "👆 Search for a company, select a popular company, "
        "or enter a Yahoo Finance ticker to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Dynamic Stock Market Price Prediction System • "
    "Yahoo Finance • Vanilla LSTM • LSTM + Temporal Attention"
)