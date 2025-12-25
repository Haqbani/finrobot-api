# FinRobot API

AI-powered Equity Research and Financial Analysis API built with FastAPI.

## Features

- 📊 Real-time stock data from Finnhub & Yahoo Finance
- 📈 Financial metrics and ratios
- 📰 Company news aggregation
- 🤖 AI-powered stock analysis
- 📋 Equity research report generation
- ⚖️ Multi-stock comparison

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check & API info |
| `/docs` | GET | Swagger UI documentation |
| `/api/v1/stock/{symbol}/profile` | GET | Company profile |
| `/api/v1/stock/{symbol}/financials` | GET | Financial metrics |
| `/api/v1/stock/{symbol}/news` | GET | Recent news |
| `/api/v1/stock/{symbol}/price` | GET | Price data |
| `/api/v1/analyze` | POST | AI stock analysis |
| `/api/v1/report` | POST | Generate research report |
| `/api/v1/compare` | POST | Compare multiple stocks |

## Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/finrobot-api.git
cd finrobot-api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn[standard] openai
pip install -e .

# Set environment variables
export FINNHUB_API_KEY=your-key-here
export FMP_API_KEY=your-key-here  # optional
export SEC_API_KEY=your-key-here  # optional

# Run server
python api_server.py
# or
uvicorn api_server:app --reload
```

### Docker

```bash
docker build -t finrobot-api .
docker run -p 8000:8000 \
  -e FINNHUB_API_KEY=your-key \
  finrobot-api
```

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

1. Fork this repository
2. Connect to Railway
3. Add environment variables:
   - `FINNHUB_API_KEY` (required)
   - `FMP_API_KEY` (optional)
   - `SEC_API_KEY` (optional)
4. Deploy!

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FINNHUB_API_KEY` | ✅ Yes | Finnhub API key for market data |
| `FMP_API_KEY` | ❌ No | Financial Modeling Prep API |
| `SEC_API_KEY` | ❌ No | SEC API for 10-K reports |
| `OPENAI_API_KEY` | ❌ No | For advanced AI analysis |

## Usage Examples

### cURL

```bash
# Get stock profile
curl https://your-api.railway.app/api/v1/stock/NVDA/profile

# Get financials
curl https://your-api.railway.app/api/v1/stock/AAPL/financials

# Analyze a stock
curl -X POST https://your-api.railway.app/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "NVDA"}'

# Compare stocks
curl -X POST https://your-api.railway.app/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT", "GOOGL"]}'
```

### Python

```python
import requests

BASE_URL = "https://your-api.railway.app"

# Get financials
response = requests.get(f"{BASE_URL}/api/v1/stock/NVDA/financials")
print(response.json())

# Analyze stock
response = requests.post(f"{BASE_URL}/api/v1/analyze", json={"symbol": "NVDA"})
analysis = response.json()
print(f"Rating: {analysis['rating']}")
print(f"Score: {analysis['score']}")
```

### JavaScript

```javascript
// Get stock price
const response = await fetch('https://your-api.railway.app/api/v1/stock/AAPL/price');
const data = await response.json();
console.log(`Current price: $${data.current_price}`);

// Generate report
const report = await fetch('https://your-api.railway.app/api/v1/report', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ symbol: 'NVDA', company_name: 'NVIDIA Corporation' })
});
console.log(await report.json());
```

## License

Apache-2.0

## Credits

Based on [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) by AI4Finance Foundation.
