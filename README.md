# AI-Based Crypto Decision Support System

Project title:
`An Intelligent Multi-Model Crypto Decision Support System using Machine Learning and Deep Learning Techniques`

Current implementation phase:
- Flask API backend + React/Tailwind dashboard
- Module 1 complete: Market Analysis
- Module 2 complete: Price Prediction (Regression + Forecasting)
- Module 3 complete: Direction Prediction (Classification)
- Module 4 complete: Sentiment Analysis (Text NLP)
- Module 5 complete: Risk Clustering (Coin Risk Segmentation)
- Module 6 complete: Image Intelligence (OCR + Sentiment + LLM Explanation)

## Architecture

- `backend/` Flask API
- `frontend/` React + Tailwind dashboard (side tabs)
- `dataset-1/` 23 coin CSV files

## Run Locally

### 1. Backend (Flask)

```powershell
pip install -r requirements.txt
python backend/app.py
```

Backend URL:
- `http://127.0.0.1:5000`

### 2. Frontend (React + Tailwind)

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL:
- `http://127.0.0.1:5173`

## DS-1 Dataset Requirements (for Module 1)

CSV file with these required columns:
- `Date` (datetime accepted, e.g., `2013-04-29 23:59:59`)
- `Name` or `Coin` (your files use `Name`)
- `Open` (numeric)
- `High` (numeric)
- `Low` (numeric)
- `Close` (numeric)
- `Volume` (numeric)

Optional columns (supported):
- `Symbol`
- `Marketcap`
- `SNo`

Recommended quality:
- At least 6 months of daily data per coin
- At least 5 coins for meaningful cross-coin analysis
- No duplicated rows for same `Coin/Name + Date`

## Module 1 Outputs

- Coin-wise mean and variance of close price
- Trend slope per coin
- Price-volume correlation per coin
- Market volatility condition label
- Volatility ranking chart
- Trend chart and correlation chart
- Responsive metrics table

## DS-2 Dataset Requirements (for Module 2)

CSV in `dataset-2/` with these required columns:
- `Timestamp` (Unix seconds)
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

## Module 2 Outputs

- Multi-model forecasting pipeline on Bitcoin close prices
- Models:
	- Linear Regression trend model
	- Autoregressive model with lag features (AR-7)
	- Holt Linear Trend smoothing forecast
- Holdout evaluation metrics: MAE, RMSE, MAPE
- Best model auto-selection by RMSE
- Future forecast with uncertainty band
- UI controls: horizon, lookback, daily/weekly frequency

## API Endpoints

- `GET /api/market-analysis?days=365`
- `GET /api/price-prediction?horizon=7&lookback=365&frequency=D`
- `GET /api/direction-prediction?lookback=2000&threshold=0.5&horizon=7`
- `POST /api/sentiment-analysis`
- `GET /api/risk-clustering?days=365&seed=42`
- `POST /api/chatbot`
- `POST /api/image-analysis` (multipart form-data with `image`, optional `question`)

Request body:

```json
{
	"text": "Bitcoin adoption is rising and market looks bullish"
}
```

Chatbot request body:

```json
{
	"message": "What does high risk cluster mean?",
	"history": [
		{"role": "user", "content": "Explain module 5"},
		{"role": "assistant", "content": "..."}
	]
}
```

Ollama runtime configuration (optional):
- `OLLAMA_HOST` (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default: `llama3`)
- `OLLAMA_VISION_MODEL` (default: `llava`)
- `OLLAMA_TIMEOUT_SECONDS` (default: `90`)

## DS-3 Dataset Requirements (for Module 3)

Place CSV in `dataset-3/` with these required columns:
- `Date` or `Timestamp` (Unix seconds)
- `Open`
- `High`
- `Low`
- `Close`

Optional but recommended:
- `Volume`

Note:
- You can directly reuse `dataset-2/btc-historical_price.csv` for Module 3.
- Current backend default for Module 3 points to `dataset-2` unless `DATASET3_DIR` is set.

Rules:
- Rows sorted by time (or sortable by Date/Timestamp)
- At least 300 rows (recommended 1000+)
- One row per candle/time-step
- Numeric OHLC values

## Module 3 Outputs

- Binary direction classes: `UP` or `DOWN`
- Models:
	- Logistic Regression (implemented from scratch with gradient descent)
	- KNN baseline (k=11)
- Evaluation metrics:
	- Accuracy
	- Precision
	- Recall
	- F1 score
- Confusion matrix (TP, TN, FP, FN)
- Probability curve on holdout set
- Feature importance view (from logistic weights)
- Future directional signal table with confidence

## DS-4 Dataset Requirements (for Module 4)

- Supervised dataset is now used from `dataset-sentiment/`.
- Current file: `dataset-sentiment/crypto_sentiment_dataset.csv`
- Required columns:
	- `text`
	- `label` (`Positive`, `Neutral`, `Negative`)
- Input text at runtime is used as inference/testing after training.

## Module 4 Outputs

- Sentiment label: Positive / Neutral / Negative
- Sentiment score and confidence
- 3-model text classification and best-model selection:
	- Multinomial Naive Bayes
	- TF-IDF Centroid Classifier
	- Lexicon Weighted (hybrid baseline)
	- Lexicon Weighted (hybrid baseline)
- Sentence-level sentiment timeline
- Aspect sentiment (Adoption, Regulation, Technology, Market Mood, Risk)
- Keyword impact ranking
- Decision-support recommendation text

## Module 6 Outputs

- OCR text extraction from uploaded image using Ollama vision model
- Routes extracted text into Module 4 sentiment pipeline
- Human-readable LLM explanation of extracted text + sentiment result
- Optional question answering over extracted text

## DS-5 Dataset Requirements (for Module 5)

- You can use the same `dataset-1` multi-coin dataset (recommended and already wired).
- No additional dataset is required for baseline risk clustering.

Required columns per coin file:
- `Date`
- `Name` (or `Coin`)
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Optional but useful:
- `Marketcap`

## Module 5 Outputs

- K-means clustering over coin-level risk features
- 3 interpreted groups:
	- Low Risk
	- Medium Risk
	- High Risk
- Coin-wise risk assignment table
- Risk index ranking
- Volatility vs drawdown cluster map
- Cluster distribution and summary insights

## Next Step (Planned)

- Final: Unified Decision Engine (combine modules 2, 3, 4, 5 into BUY/HOLD/SELL with risk note)
