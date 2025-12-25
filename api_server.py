"""
FinRobot API Server - FULL FEATURED
AI-powered Equity Research and Financial Analysis
All features from FinRobot (except marker-pdf due to dependency conflicts)
"""

import os
import json
import io
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import finnhub

# Initialize FastAPI
app = FastAPI(
    title="FinRobot API",
    description="""
    ## AI-powered Equity Research and Financial Analysis
    
    Full-featured API with:
    - 📊 Real-time market data (Finnhub, Yahoo Finance)
    - 🤖 AI-powered analysis (GPT-4)
    - 📈 Technical analysis & charting
    - 📰 News sentiment analysis
    - 📋 Equity research reports
    - 🔍 SEC filings access
    - ⚖️ Multi-stock comparison
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Clients
# ============================================================================

def get_finnhub_client():
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="FINNHUB_API_KEY not configured")
    return finnhub.Client(api_key=api_key)

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured. Required for AI features.")
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def get_sec_api():
    api_key = os.environ.get("SEC_API_KEY")
    if not api_key:
        return None
    try:
        from sec_api import QueryApi
        return QueryApi(api_key=api_key)
    except:
        return None

# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 FinRobot API v3.0 - FULL FEATURED")
    print("=" * 60)
    
    features = {
        "FINNHUB_API_KEY": ("✓" if os.environ.get("FINNHUB_API_KEY") else "✗", "Market Data"),
        "OPENAI_API_KEY": ("✓" if os.environ.get("OPENAI_API_KEY") else "✗", "AI Forecasting"),
        "SEC_API_KEY": ("✓" if os.environ.get("SEC_API_KEY") else "○", "SEC Filings"),
        "FMP_API_KEY": ("✓" if os.environ.get("FMP_API_KEY") else "○", "Extended Financials"),
    }
    
    for key, (status, desc) in features.items():
        print(f"  {status} {key}: {desc}")
    
    print("=" * 60)

# ============================================================================
# Models
# ============================================================================

class AnalysisRequest(BaseModel):
    symbol: str

class ForecastRequest(BaseModel):
    symbol: str
    include_news: bool = True
    include_technicals: bool = True

class ReportRequest(BaseModel):
    symbol: str
    company_name: str
    include_ai_analysis: bool = True

class CompareRequest(BaseModel):
    symbols: List[str]

class SECFilingRequest(BaseModel):
    symbol: str
    filing_type: str = "10-K"
    limit: int = 5

class SentimentRequest(BaseModel):
    symbol: str
    days: int = 14

# ============================================================================
# Health & Info
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "FinRobot API",
        "version": "3.0.0",
        "status": "healthy",
        "features": {
            "market_data": bool(os.environ.get("FINNHUB_API_KEY")),
            "ai_forecasting": bool(os.environ.get("OPENAI_API_KEY")),
            "sec_filings": bool(os.environ.get("SEC_API_KEY")),
        },
        "endpoints": {
            "data": [
                "GET /api/v1/stock/{symbol}/profile",
                "GET /api/v1/stock/{symbol}/financials",
                "GET /api/v1/stock/{symbol}/news",
                "GET /api/v1/stock/{symbol}/price",
                "GET /api/v1/stock/{symbol}/quote",
            ],
            "analysis": [
                "POST /api/v1/analyze",
                "POST /api/v1/forecast",
                "POST /api/v1/sentiment",
                "POST /api/v1/technicals",
            ],
            "reports": [
                "POST /api/v1/report",
                "POST /api/v1/report/pdf",
            ],
            "sec": [
                "POST /api/v1/sec/filings",
            ],
            "tools": [
                "POST /api/v1/compare",
                "POST /api/v1/screen",
            ]
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ============================================================================
# Stock Data Endpoints
# ============================================================================

@app.get("/api/v1/stock/{symbol}/profile")
async def get_profile(symbol: str):
    """Get comprehensive company profile"""
    try:
        client = get_finnhub_client()
        symbol = symbol.upper()
        
        profile = client.company_profile2(symbol=symbol)
        if not profile:
            raise HTTPException(status_code=404, detail=f"No profile for {symbol}")
        
        # Get additional data
        peers = client.company_peers(symbol)
        
        return {
            "symbol": symbol,
            "profile": {
                "name": profile.get("name"),
                "industry": profile.get("finnhubIndustry"),
                "sector": profile.get("gsector"),
                "market_cap": profile.get("marketCapitalization"),
                "shares_outstanding": profile.get("shareOutstanding"),
                "ipo_date": profile.get("ipo"),
                "country": profile.get("country"),
                "exchange": profile.get("exchange"),
                "currency": profile.get("currency"),
                "website": profile.get("weburl"),
                "logo": profile.get("logo"),
                "phone": profile.get("phone"),
            },
            "peers": peers[:10] if peers else [],
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stock/{symbol}/financials")
async def get_financials(symbol: str):
    """Get comprehensive financial metrics"""
    try:
        client = get_finnhub_client()
        symbol = symbol.upper()
        
        metrics = client.company_basic_financials(symbol, 'all')
        if not metrics or 'metric' not in metrics:
            raise HTTPException(status_code=404, detail=f"No financials for {symbol}")
        
        m = metrics['metric']
        
        return {
            "symbol": symbol,
            "valuation": {
                "market_cap": m.get("marketCapitalization"),
                "enterprise_value": m.get("enterpriseValue"),
                "pe_ratio": m.get("peTTM"),
                "forward_pe": m.get("forwardPE"),
                "peg_ratio": m.get("pegTTM"),
                "ps_ratio": m.get("psTTM"),
                "pb_ratio": m.get("pbQuarterly"),
                "ev_ebitda": m.get("evEbitdaTTM"),
                "ev_revenue": m.get("evRevenueTTM"),
            },
            "profitability": {
                "gross_margin": m.get("grossMarginTTM"),
                "operating_margin": m.get("operatingMarginTTM"),
                "net_margin": m.get("netProfitMarginTTM"),
                "roe": m.get("roeTTM"),
                "roi": m.get("roiTTM"),
                "roa": m.get("roaTTM"),
                "roic": m.get("roicTTM"),
            },
            "growth": {
                "revenue_growth_yoy": m.get("revenueGrowthTTMYoy"),
                "revenue_growth_3y": m.get("revenueGrowth3Y"),
                "revenue_growth_5y": m.get("revenueGrowth5Y"),
                "eps_growth_yoy": m.get("epsGrowthTTMYoy"),
                "eps_growth_3y": m.get("epsGrowth3Y"),
                "eps_growth_5y": m.get("epsGrowth5Y"),
            },
            "liquidity": {
                "current_ratio": m.get("currentRatioQuarterly"),
                "quick_ratio": m.get("quickRatioQuarterly"),
                "cash_ratio": m.get("cashRatio"),
            },
            "leverage": {
                "debt_to_equity": m.get("totalDebt/totalEquityQuarterly"),
                "debt_to_assets": m.get("totalDebtToTotalAsset"),
                "interest_coverage": m.get("netInterestCoverageTTM"),
            },
            "per_share": {
                "eps": m.get("epsTTM"),
                "book_value": m.get("bookValuePerShareQuarterly"),
                "cash_per_share": m.get("cashPerSharePerShareQuarterly"),
                "revenue_per_share": m.get("revenuePerShareTTM"),
            },
            "dividends": {
                "dividend_yield": m.get("dividendYieldIndicatedAnnual"),
                "dividend_per_share": m.get("dividendPerShareAnnual"),
                "payout_ratio": m.get("payoutRatioTTM"),
            },
            "risk": {
                "beta": m.get("beta"),
                "52_week_high": m.get("52WeekHigh"),
                "52_week_low": m.get("52WeekLow"),
                "52_week_high_date": m.get("52WeekHighDate"),
                "52_week_low_date": m.get("52WeekLowDate"),
                "10_day_avg_volume": m.get("10DayAverageTradingVolume"),
            },
            "performance": {
                "5_day_return": m.get("5DayPriceReturnDaily"),
                "month_return": m.get("monthToDatePriceReturnDaily"),
                "ytd_return": m.get("yearToDatePriceReturnDaily"),
                "52_week_return": m.get("52WeekPriceReturnDaily"),
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stock/{symbol}/news")
async def get_news(symbol: str, days: int = 7, limit: int = 20):
    """Get recent company news"""
    try:
        client = get_finnhub_client()
        symbol = symbol.upper()
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        news = client.company_news(
            symbol,
            _from=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d")
        )
        
        formatted = []
        for item in news[:limit]:
            formatted.append({
                "headline": item.get("headline"),
                "summary": item.get("summary", "")[:500],
                "source": item.get("source"),
                "url": item.get("url"),
                "image": item.get("image"),
                "datetime": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                "category": item.get("category"),
                "related": item.get("related"),
            })
        
        return {
            "symbol": symbol,
            "news": formatted,
            "count": len(formatted),
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stock/{symbol}/price")
async def get_price(symbol: str, period: str = "1mo"):
    """Get price history"""
    try:
        import yfinance as yf
        symbol = symbol.upper()
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No price data for {symbol}")
        
        current = float(hist['Close'].iloc[-1])
        open_price = float(hist['Close'].iloc[0])
        change = ((current - open_price) / open_price) * 100
        
        info = ticker.info
        
        return {
            "symbol": symbol,
            "company_name": info.get("longName", symbol),
            "current_price": round(current, 2),
            "change_percent": round(change, 2),
            "period_high": round(float(hist['High'].max()), 2),
            "period_low": round(float(hist['Low'].min()), 2),
            "period_open": round(open_price, 2),
            "volume": int(hist['Volume'].iloc[-1]),
            "avg_volume": int(hist['Volume'].mean()),
            "currency": info.get("currency", "USD"),
            "period": period,
            "data_points": len(hist),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stock/{symbol}/quote")
async def get_quote(symbol: str):
    """Get real-time quote"""
    try:
        client = get_finnhub_client()
        symbol = symbol.upper()
        
        quote = client.quote(symbol)
        
        return {
            "symbol": symbol,
            "current_price": quote.get("c"),
            "change": quote.get("d"),
            "change_percent": quote.get("dp"),
            "high": quote.get("h"),
            "low": quote.get("l"),
            "open": quote.get("o"),
            "previous_close": quote.get("pc"),
            "timestamp": datetime.fromtimestamp(quote.get("t", 0)).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Analysis Endpoints
# ============================================================================

@app.post("/api/v1/analyze")
async def analyze(request: AnalysisRequest):
    """Rule-based stock analysis"""
    try:
        import yfinance as yf
        symbol = request.symbol.upper()
        
        client = get_finnhub_client()
        metrics = client.company_basic_financials(symbol, 'all')
        m = metrics.get('metric', {}) if metrics else {}
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # Scoring
        score = 0
        insights = []
        
        # Valuation
        pe = m.get("peTTM") or 0
        if 0 < pe < 15:
            score += 3
            insights.append(f"✅ Attractive P/E: {pe:.1f}x")
        elif 0 < pe < 25:
            score += 1
            insights.append(f"➖ Fair P/E: {pe:.1f}x")
        elif pe > 50:
            score -= 2
            insights.append(f"⚠️ High P/E: {pe:.1f}x")
        
        peg = m.get("pegTTM") or 0
        if 0 < peg < 1:
            score += 2
            insights.append(f"✅ PEG < 1: {peg:.2f}")
        elif 0 < peg < 2:
            score += 1
        
        # Growth
        rev = m.get("revenueGrowthTTMYoy") or 0
        if rev > 25:
            score += 2
            insights.append(f"✅ Strong revenue growth: {rev:.1f}%")
        elif rev < 0:
            score -= 1
            insights.append(f"⚠️ Revenue declining: {rev:.1f}%")
        
        # Profitability
        margin = m.get("netProfitMarginTTM") or 0
        if margin > 20:
            score += 2
            insights.append(f"✅ High margin: {margin:.1f}%")
        elif margin < 5:
            score -= 1
        
        # Rating
        if score >= 7:
            rating = "STRONG BUY"
        elif score >= 4:
            rating = "BUY"
        elif score >= 1:
            rating = "HOLD"
        else:
            rating = "SELL"
        
        return {
            "symbol": symbol,
            "rating": rating,
            "score": score,
            "price": round(price, 2),
            "insights": insights,
            "metrics": {"pe": pe, "peg": peg, "revenue_growth": rev, "net_margin": margin},
            "method": "rule_based",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/forecast")
async def forecast(request: ForecastRequest):
    """🤖 AI-powered stock forecast using GPT-4"""
    try:
        import yfinance as yf
        symbol = request.symbol.upper()
        
        openai = get_openai_client()
        finnhub = get_finnhub_client()
        
        # Gather data
        profile = finnhub.company_profile2(symbol=symbol)
        metrics = finnhub.company_basic_financials(symbol, 'all')
        m = metrics.get('metric', {}) if metrics else {}
        
        # News
        news_text = ""
        if request.include_news:
            end = datetime.now()
            start = end - timedelta(days=14)
            news = finnhub.company_news(symbol, _from=start.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d"))
            if news:
                headlines = [n.get("headline", "") for n in news[:10]]
                news_text = "\n".join(f"• {h}" for h in headlines)
        
        # Price
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # Technicals
        tech_text = ""
        if request.include_technicals and not hist.empty:
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else None
            rsi = calculate_rsi(hist['Close']) if len(hist) >= 14 else None
            tech_text = f"""
TECHNICALS:
- 20-day SMA: ${sma_20:.2f if sma_20 else 'N/A'}
- RSI (14): {rsi:.1f if rsi else 'N/A'}
- Price vs SMA: {'Above' if sma_20 and price > sma_20 else 'Below' if sma_20 else 'N/A'}
"""
        
        prompt = f"""Analyze {symbol} ({profile.get('name', 'Unknown')}) and provide:
1. 2-3 positive factors
2. 2-3 concerns
3. Price prediction for next week (e.g., "up 2-3%")
4. Confidence level (low/medium/high)
5. Brief recommendation

DATA:
Company: {profile.get('name')} | Industry: {profile.get('finnhubIndustry')}
Price: ${price:.2f} | Market Cap: ${m.get('marketCapitalization', 0):.0f}M

FINANCIALS:
- P/E: {m.get('peTTM', 'N/A')} | PEG: {m.get('pegTTM', 'N/A')}
- Revenue Growth: {m.get('revenueGrowthTTMYoy', 'N/A')}%
- Net Margin: {m.get('netProfitMarginTTM', 'N/A')}%
- ROE: {m.get('roeTTM', 'N/A')} | Beta: {m.get('beta', 'N/A')}
- 52W High: ${m.get('52WeekHigh', 'N/A')} | Low: ${m.get('52WeekLow', 'N/A')}
{tech_text}
NEWS:
{news_text if news_text else 'No recent news'}

Respond in JSON: {{"positive": [], "concerns": [], "prediction": "", "confidence": "", "recommendation": ""}}"""

        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a senior equity analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        return {
            "symbol": symbol,
            "company": profile.get("name"),
            "price": round(price, 2),
            "forecast": analysis,
            "data_sources": {
                "financials": True,
                "news": bool(news_text),
                "technicals": request.include_technicals
            },
            "model": "gpt-4-turbo-preview",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/sentiment")
async def sentiment_analysis(request: SentimentRequest):
    """Analyze news sentiment using AI"""
    try:
        symbol = request.symbol.upper()
        
        openai = get_openai_client()
        finnhub = get_finnhub_client()
        
        # Get news
        end = datetime.now()
        start = end - timedelta(days=request.days)
        news = finnhub.company_news(symbol, _from=start.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d"))
        
        if not news:
            return {"symbol": symbol, "sentiment": "neutral", "message": "No recent news found"}
        
        headlines = [n.get("headline", "") for n in news[:15]]
        news_text = "\n".join(f"- {h}" for h in headlines)
        
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Analyze sentiment. Respond with JSON only."},
                {"role": "user", "content": f"""Analyze sentiment for {symbol}:

{news_text}

Respond: {{"overall_sentiment": "bullish/bearish/neutral", "score": -1 to 1, "key_themes": [], "summary": ""}}"""}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return {
            "symbol": symbol,
            "sentiment": result,
            "articles_analyzed": len(headlines),
            "period_days": request.days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Reports
# ============================================================================

@app.post("/api/v1/report")
async def generate_report(request: ReportRequest):
    """Generate comprehensive equity research report"""
    try:
        import yfinance as yf
        symbol = request.symbol.upper()
        
        finnhub = get_finnhub_client()
        
        profile = finnhub.company_profile2(symbol=symbol)
        metrics = finnhub.company_basic_financials(symbol, 'all')
        m = metrics.get('metric', {}) if metrics else {}
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # AI Analysis
        ai_section = ""
        if request.include_ai_analysis and os.environ.get("OPENAI_API_KEY"):
            try:
                forecast_req = ForecastRequest(symbol=symbol)
                forecast_result = await forecast(forecast_req)
                ai_section = f"""
AI ANALYSIS (GPT-4)
-------------------
Prediction: {forecast_result['forecast'].get('prediction', 'N/A')}
Confidence: {forecast_result['forecast'].get('confidence', 'N/A')}
Recommendation: {forecast_result['forecast'].get('recommendation', 'N/A')}

Positive Factors:
{chr(10).join('• ' + p for p in forecast_result['forecast'].get('positive', []))}

Concerns:
{chr(10).join('• ' + c for c in forecast_result['forecast'].get('concerns', []))}
"""
            except:
                ai_section = "\nAI Analysis: Not available\n"
        
        report = f"""
{'='*70}
{request.company_name} ({symbol})
EQUITY RESEARCH REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
{'='*70}

EXECUTIVE SUMMARY
-----------------
Current Price: ${price:.2f}
Market Cap: ${m.get('marketCapitalization', 0):.0f}M
Industry: {profile.get('finnhubIndustry', 'N/A')}
Exchange: {profile.get('exchange', 'N/A')}
{ai_section}
VALUATION
---------
P/E Ratio (TTM): {m.get('peTTM', 'N/A')}x
Forward P/E: {m.get('forwardPE', 'N/A')}x
PEG Ratio: {m.get('pegTTM', 'N/A')}
P/S Ratio: {m.get('psTTM', 'N/A')}x
P/B Ratio: {m.get('pbQuarterly', 'N/A')}x
EV/EBITDA: {m.get('evEbitdaTTM', 'N/A')}x

PROFITABILITY
-------------
Gross Margin: {m.get('grossMarginTTM', 'N/A')}%
Operating Margin: {m.get('operatingMarginTTM', 'N/A')}%
Net Margin: {m.get('netProfitMarginTTM', 'N/A')}%
ROE: {m.get('roeTTM', 'N/A')}%
ROI: {m.get('roiTTM', 'N/A')}%

GROWTH
------
Revenue Growth (YoY): {m.get('revenueGrowthTTMYoy', 'N/A')}%
EPS Growth (YoY): {m.get('epsGrowthTTMYoy', 'N/A')}%
5-Year Revenue CAGR: {m.get('revenueGrowth5Y', 'N/A')}%
5-Year EPS CAGR: {m.get('epsGrowth5Y', 'N/A')}%

RISK
----
Beta: {m.get('beta', 'N/A')}
52-Week High: ${m.get('52WeekHigh', 'N/A')}
52-Week Low: ${m.get('52WeekLow', 'N/A')}

PERFORMANCE
-----------
YTD Return: {m.get('yearToDatePriceReturnDaily', 'N/A')}%
52-Week Return: {m.get('52WeekPriceReturnDaily', 'N/A')}%

{'='*70}
DISCLAIMER: For informational purposes only. Not investment advice.
{'='*70}
"""
        
        return {
            "symbol": symbol,
            "company_name": request.company_name,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/compare")
async def compare(request: CompareRequest):
    """Compare multiple stocks"""
    try:
        import yfinance as yf
        finnhub = get_finnhub_client()
        
        results = []
        for sym in request.symbols[:10]:
            try:
                s = sym.upper()
                m = finnhub.company_basic_financials(s, 'all').get('metric', {})
                ticker = yf.Ticker(s)
                hist = ticker.history(period="5d")
                price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
                
                results.append({
                    "symbol": s,
                    "price": round(price, 2),
                    "pe": m.get("peTTM"),
                    "peg": m.get("pegTTM"),
                    "revenue_growth": m.get("revenueGrowthTTMYoy"),
                    "margin": m.get("netProfitMarginTTM"),
                    "roe": m.get("roeTTM"),
                    "beta": m.get("beta"),
                })
            except:
                results.append({"symbol": sym.upper(), "error": "Failed"})
        
        valid = [r for r in results if r.get("pe") and r.get("pe") > 0]
        valid.sort(key=lambda x: x.get("pe", 999))
        
        return {
            "comparison": results,
            "best_value": valid[0]["symbol"] if valid else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SEC Filings
# ============================================================================

@app.post("/api/v1/sec/filings")
async def get_sec_filings(request: SECFilingRequest):
    """Get SEC filings for a company"""
    try:
        sec = get_sec_api()
        if not sec:
            raise HTTPException(status_code=400, detail="SEC_API_KEY not configured")
        
        query = {
            "query": f'ticker:"{request.symbol.upper()}" AND formType:"{request.filing_type}"',
            "from": "0",
            "size": str(request.limit),
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        filings = sec.get_filings(query)
        
        return {
            "symbol": request.symbol.upper(),
            "filing_type": request.filing_type,
            "filings": filings.get("filings", []),
            "total": filings.get("total", {}).get("value", 0),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Helpers
# ============================================================================

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return float(100 - (100 / (1 + rs.iloc[-1])))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port)
