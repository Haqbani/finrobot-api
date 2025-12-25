"""
FinRobot API Server - Full Featured (Railway Compatible)
AI-powered Equity Research and Financial Analysis
"""

import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import finnhub
import asyncio

# Initialize FastAPI
app = FastAPI(
    title="FinRobot API",
    description="AI-powered Equity Research and Financial Analysis API with GPT-4",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Clients & Config
# ============================================================================

def get_finnhub_client():
    """Get Finnhub client"""
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="FINNHUB_API_KEY not configured")
    return finnhub.Client(api_key=api_key)

def get_openai_client():
    """Get OpenAI client"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    from openai import OpenAI
    return OpenAI(api_key=api_key)

# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 50)
    print("FinRobot API v2.0 - Full Featured")
    print("=" * 50)
    
    keys = {
        "FINNHUB_API_KEY": ("required", "Market data"),
        "OPENAI_API_KEY": ("required for AI", "GPT-4 forecasting"),
        "FMP_API_KEY": ("optional", "Extended financials"),
    }
    
    for key, (status, desc) in keys.items():
        if os.environ.get(key):
            print(f"✓ {key} - {desc}")
        else:
            symbol = "✗" if "required" in status else "○"
            print(f"{symbol} {key} not set ({status}) - {desc}")
    
    print("=" * 50)

# ============================================================================
# Request/Response Models
# ============================================================================

class StockAnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "comprehensive"

class AIForecastRequest(BaseModel):
    symbol: str
    include_news: bool = True

class ReportRequest(BaseModel):
    symbol: str
    company_name: str

class CompareRequest(BaseModel):
    symbols: List[str]

# ============================================================================
# Health Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "status": "healthy",
        "service": "FinRobot API",
        "version": "2.0.0",
        "features": {
            "ai_forecasting": bool(os.environ.get("OPENAI_API_KEY")),
            "market_data": bool(os.environ.get("FINNHUB_API_KEY")),
        },
        "docs": "/docs",
        "endpoints": {
            "stock_profile": "GET /api/v1/stock/{symbol}/profile",
            "stock_financials": "GET /api/v1/stock/{symbol}/financials",
            "stock_news": "GET /api/v1/stock/{symbol}/news",
            "stock_price": "GET /api/v1/stock/{symbol}/price",
            "analyze": "POST /api/v1/analyze",
            "ai_forecast": "POST /api/v1/forecast (GPT-4)",
            "report": "POST /api/v1/report",
            "compare": "POST /api/v1/compare"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

# ============================================================================
# Stock Data Endpoints
# ============================================================================

@app.get("/api/v1/stock/{symbol}/profile")
async def get_stock_profile(symbol: str):
    """Get company profile"""
    try:
        client = get_finnhub_client()
        profile = client.company_profile2(symbol=symbol.upper())
        
        if not profile:
            raise HTTPException(status_code=404, detail=f"No profile found for {symbol}")
        
        return {
            "symbol": symbol.upper(),
            "name": profile.get("name"),
            "industry": profile.get("finnhubIndustry"),
            "market_cap": profile.get("marketCapitalization"),
            "ipo_date": profile.get("ipo"),
            "country": profile.get("country"),
            "exchange": profile.get("exchange"),
            "currency": profile.get("currency"),
            "website": profile.get("weburl"),
            "logo": profile.get("logo"),
            "raw": profile,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stock/{symbol}/financials")
async def get_stock_financials(symbol: str):
    """Get detailed financial metrics"""
    try:
        client = get_finnhub_client()
        metrics = client.company_basic_financials(symbol.upper(), 'all')
        
        if not metrics or 'metric' not in metrics:
            raise HTTPException(status_code=404, detail=f"No financials found for {symbol}")
        
        m = metrics['metric']
        
        return {
            "symbol": symbol.upper(),
            "valuation": {
                "pe_ratio": m.get("peTTM"),
                "forward_pe": m.get("forwardPE"),
                "ps_ratio": m.get("psTTM"),
                "pb_ratio": m.get("pbQuarterly"),
                "ev_ebitda": m.get("evEbitdaTTM"),
                "peg_ratio": m.get("pegTTM"),
                "market_cap": m.get("marketCapitalization"),
            },
            "profitability": {
                "gross_margin": m.get("grossMarginTTM"),
                "operating_margin": m.get("operatingMarginTTM"),
                "net_margin": m.get("netProfitMarginTTM"),
                "roe": m.get("roeTTM"),
                "roi": m.get("roiTTM"),
                "roa": m.get("roaTTM"),
            },
            "growth": {
                "revenue_growth_yoy": m.get("revenueGrowthTTMYoy"),
                "eps_growth_yoy": m.get("epsGrowthTTMYoy"),
                "revenue_growth_5y": m.get("revenueGrowth5Y"),
                "eps_growth_5y": m.get("epsGrowth5Y"),
            },
            "risk": {
                "beta": m.get("beta"),
                "52_week_high": m.get("52WeekHigh"),
                "52_week_low": m.get("52WeekLow"),
                "52_week_high_date": m.get("52WeekHighDate"),
                "52_week_low_date": m.get("52WeekLowDate"),
            },
            "performance": {
                "ytd_return": m.get("yearToDatePriceReturnDaily"),
                "52_week_return": m.get("52WeekPriceReturnDaily"),
                "5_day_return": m.get("5DayPriceReturnDaily"),
                "month_return": m.get("monthToDatePriceReturnDaily"),
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stock/{symbol}/news")
async def get_stock_news(symbol: str, days: int = 7):
    """Get recent company news"""
    try:
        client = get_finnhub_client()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        news = client.company_news(
            symbol.upper(),
            _from=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d")
        )
        
        formatted_news = []
        for item in news[:20]:
            formatted_news.append({
                "headline": item.get("headline"),
                "summary": item.get("summary", "")[:500],
                "source": item.get("source"),
                "url": item.get("url"),
                "datetime": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                "category": item.get("category"),
            })
        
        return {
            "symbol": symbol.upper(),
            "news": formatted_news,
            "count": len(formatted_news),
            "period": f"Last {days} days",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stock/{symbol}/price")
async def get_stock_price(symbol: str, period: str = "1mo"):
    """Get stock price data"""
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")
        
        current = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[0])
        change_pct = ((current - prev) / prev) * 100
        
        info = ticker.info
        
        return {
            "symbol": symbol.upper(),
            "company_name": info.get("longName", symbol.upper()),
            "current_price": round(current, 2),
            "period_start_price": round(prev, 2),
            "change_percent": round(change_pct, 2),
            "period_high": round(float(hist['High'].max()), 2),
            "period_low": round(float(hist['Low'].min()), 2),
            "avg_volume": int(hist['Volume'].mean()),
            "currency": info.get("currency", "USD"),
            "period": period,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AI-Powered Analysis Endpoints
# ============================================================================

@app.post("/api/v1/analyze")
async def analyze_stock(request: StockAnalysisRequest):
    """Rule-based stock analysis (no AI required)"""
    try:
        import yfinance as yf
        
        symbol = request.symbol.upper()
        client = get_finnhub_client()
        
        # Get data
        metrics = client.company_basic_financials(symbol, 'all')
        m = metrics.get('metric', {}) if metrics else {}
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # Score calculation
        score = 0
        insights = []
        
        pe = m.get("peTTM") or 0
        if 0 < pe < 15:
            score += 3
            insights.append(f"Attractive P/E: {pe:.1f}x (below 15)")
        elif 0 < pe < 25:
            score += 2
            insights.append(f"Fair P/E: {pe:.1f}x")
        elif pe > 50:
            score -= 1
            insights.append(f"High P/E: {pe:.1f}x (premium valuation)")
        
        peg = m.get("pegTTM") or 0
        if 0 < peg < 1:
            score += 2
            insights.append(f"PEG {peg:.2f} < 1 (undervalued vs growth)")
        
        rev_growth = m.get("revenueGrowthTTMYoy") or 0
        if rev_growth > 30:
            score += 2
            insights.append(f"Strong revenue growth: {rev_growth:.1f}%")
        elif rev_growth > 10:
            score += 1
            insights.append(f"Solid revenue growth: {rev_growth:.1f}%")
        elif rev_growth < 0:
            score -= 1
            insights.append(f"Revenue declining: {rev_growth:.1f}%")
        
        net_margin = m.get("netProfitMarginTTM") or 0
        if net_margin > 20:
            score += 2
            insights.append(f"High profitability: {net_margin:.1f}% margin")
        elif net_margin > 10:
            score += 1
        
        # Rating
        if score >= 8:
            rating = "STRONG BUY"
        elif score >= 5:
            rating = "BUY"
        elif score >= 2:
            rating = "HOLD"
        else:
            rating = "UNDERPERFORM"
        
        return {
            "symbol": symbol,
            "rating": rating,
            "score": score,
            "current_price": round(current_price, 2),
            "insights": insights,
            "metrics": {
                "pe_ratio": pe,
                "peg_ratio": peg,
                "revenue_growth": rev_growth,
                "net_margin": net_margin,
                "beta": m.get("beta"),
            },
            "method": "rule_based",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/forecast")
async def ai_forecast(request: AIForecastRequest):
    """
    🤖 AI-Powered Stock Forecast using GPT-4
    
    Analyzes financials, news, and market data to predict stock movement.
    Requires OPENAI_API_KEY environment variable.
    """
    try:
        import yfinance as yf
        
        symbol = request.symbol.upper()
        
        # Get OpenAI client
        openai_client = get_openai_client()
        finnhub_client = get_finnhub_client()
        
        # Gather data
        profile = finnhub_client.company_profile2(symbol=symbol)
        metrics = finnhub_client.company_basic_financials(symbol, 'all')
        m = metrics.get('metric', {}) if metrics else {}
        
        # Get news if requested
        news_summary = ""
        if request.include_news:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            news = finnhub_client.company_news(
                symbol,
                _from=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            if news:
                news_headlines = [n.get("headline", "") for n in news[:10]]
                news_summary = "\n".join(f"- {h}" for h in news_headlines)
        
        # Get price data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # Build prompt for GPT-4
        prompt = f"""You are an expert financial analyst. Analyze the following data for {symbol} ({profile.get('name', 'Unknown Company')}) and provide:

1. A brief analysis of positive developments (2-3 key factors)
2. Potential concerns (2-3 key factors)
3. A price movement prediction for next week (e.g., "up 2-3%" or "down 1-2%")
4. A confidence level (low/medium/high)
5. A summary recommendation

COMPANY DATA:
- Industry: {profile.get('finnhubIndustry', 'N/A')}
- Market Cap: ${m.get('marketCapitalization', 0):.0f}M
- Current Price: ${current_price:.2f}

FINANCIAL METRICS:
- P/E Ratio: {m.get('peTTM', 'N/A')}
- PEG Ratio: {m.get('pegTTM', 'N/A')}
- Revenue Growth YoY: {m.get('revenueGrowthTTMYoy', 'N/A')}%
- EPS Growth YoY: {m.get('epsGrowthTTMYoy', 'N/A')}%
- Net Margin: {m.get('netProfitMarginTTM', 'N/A')}%
- ROE: {m.get('roeTTM', 'N/A')}
- Beta: {m.get('beta', 'N/A')}
- 52-Week High: ${m.get('52WeekHigh', 'N/A')}
- 52-Week Low: ${m.get('52WeekLow', 'N/A')}
- YTD Return: {m.get('yearToDatePriceReturnDaily', 'N/A')}%

RECENT NEWS HEADLINES:
{news_summary if news_summary else "No recent news available"}

Provide your analysis in a structured JSON format with keys: positive_factors, concerns, prediction, confidence, recommendation, summary
"""
        
        # Call GPT-4
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a professional equity research analyst. Provide analysis in valid JSON format only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        ai_analysis = json.loads(response.choices[0].message.content)
        
        return {
            "symbol": symbol,
            "company_name": profile.get("name"),
            "current_price": round(current_price, 2),
            "analysis": ai_analysis,
            "data_used": {
                "financials": True,
                "news_articles": len(news) if request.include_news and news else 0,
                "price_history": "1 month"
            },
            "model": "gpt-4-turbo-preview",
            "method": "ai_powered",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Forecast error: {str(e)}")

@app.post("/api/v1/report")
async def generate_report(request: ReportRequest):
    """Generate equity research report"""
    try:
        import yfinance as yf
        
        symbol = request.symbol.upper()
        company_name = request.company_name
        client = get_finnhub_client()
        
        # Get data
        profile = client.company_profile2(symbol=symbol)
        metrics = client.company_basic_financials(symbol, 'all')
        m = metrics.get('metric', {}) if metrics else {}
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # Build report
        report_text = f"""
================================================================================
{company_name} ({symbol})
EQUITY RESEARCH REPORT
{datetime.now().strftime("%Y-%m-%d")}
================================================================================

EXECUTIVE SUMMARY
-----------------
Current Price: ${current_price:.2f}
Market Cap: ${m.get('marketCapitalization', 0):.0f}M
Industry: {profile.get('finnhubIndustry', 'N/A')}
Exchange: {profile.get('exchange', 'N/A')}

VALUATION METRICS
-----------------
P/E Ratio (TTM): {m.get('peTTM', 'N/A')}x
Forward P/E: {m.get('forwardPE', 'N/A')}x
P/S Ratio: {m.get('psTTM', 'N/A')}x
P/B Ratio: {m.get('pbQuarterly', 'N/A')}x
EV/EBITDA: {m.get('evEbitdaTTM', 'N/A')}x
PEG Ratio: {m.get('pegTTM', 'N/A')}

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

RISK METRICS
------------
Beta: {m.get('beta', 'N/A')}
52-Week High: ${m.get('52WeekHigh', 'N/A')}
52-Week Low: ${m.get('52WeekLow', 'N/A')}

PERFORMANCE
-----------
YTD Return: {m.get('yearToDatePriceReturnDaily', 'N/A')}%
52-Week Return: {m.get('52WeekPriceReturnDaily', 'N/A')}%

================================================================================
DISCLAIMER: This report is for informational purposes only and does not 
constitute investment advice. Past performance does not guarantee future results.
================================================================================
"""
        
        return {
            "symbol": symbol,
            "company_name": company_name,
            "report_text": report_text,
            "data": {
                "current_price": round(current_price, 2),
                "valuation": {
                    "pe_ratio": m.get("peTTM"),
                    "peg_ratio": m.get("pegTTM"),
                    "ps_ratio": m.get("psTTM"),
                },
                "profitability": {
                    "gross_margin": m.get("grossMarginTTM"),
                    "net_margin": m.get("netProfitMarginTTM"),
                    "roe": m.get("roeTTM"),
                },
                "growth": {
                    "revenue_growth": m.get("revenueGrowthTTMYoy"),
                    "eps_growth": m.get("epsGrowthTTMYoy"),
                },
                "risk": {
                    "beta": m.get("beta"),
                    "52_week_high": m.get("52WeekHigh"),
                    "52_week_low": m.get("52WeekLow"),
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/compare")
async def compare_stocks(request: CompareRequest):
    """Compare multiple stocks"""
    try:
        import yfinance as yf
        
        client = get_finnhub_client()
        results = []
        
        for symbol in request.symbols[:5]:
            try:
                sym = symbol.upper()
                metrics = client.company_basic_financials(sym, 'all')
                m = metrics.get('metric', {}) if metrics else {}
                
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="1mo")
                price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
                
                results.append({
                    "symbol": sym,
                    "price": round(price, 2),
                    "pe_ratio": m.get("peTTM"),
                    "peg_ratio": m.get("pegTTM"),
                    "revenue_growth": m.get("revenueGrowthTTMYoy"),
                    "net_margin": m.get("netProfitMarginTTM"),
                    "roe": m.get("roeTTM"),
                    "beta": m.get("beta"),
                })
            except Exception as e:
                results.append({"symbol": symbol.upper(), "error": str(e)})
        
        # Find best value (lowest positive PE)
        valid = [r for r in results if r.get("pe_ratio") and r.get("pe_ratio") > 0]
        valid.sort(key=lambda x: x.get("pe_ratio", 999))
        
        return {
            "comparison": results,
            "best_value": valid[0]["symbol"] if valid else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
