"""
FinRobot API Server - Railway Optimized (Standalone)
AI-powered Equity Research and Financial Analysis
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import finnhub

# Initialize FastAPI
app = FastAPI(
    title="FinRobot API",
    description="AI-powered Equity Research and Financial Analysis API",
    version="1.0.0",
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
# Finnhub Client
# ============================================================================

def get_finnhub_client():
    """Get Finnhub client"""
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="FINNHUB_API_KEY not configured")
    return finnhub.Client(api_key=api_key)

# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 50)
    print("FinRobot API Starting...")
    print("=" * 50)
    
    keys = {
        "FINNHUB_API_KEY": "required",
        "FMP_API_KEY": "optional",
        "SEC_API_KEY": "optional",
        "OPENAI_API_KEY": "optional"
    }
    
    for key, status in keys.items():
        if os.environ.get(key):
            print(f"✓ {key} configured")
        else:
            symbol = "✗" if status == "required" else "○"
            print(f"{symbol} {key} not set ({status})")
    
    print("=" * 50)

# ============================================================================
# Request/Response Models
# ============================================================================

class StockAnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "comprehensive"

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
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "stock_profile": "/api/v1/stock/{symbol}/profile",
            "stock_financials": "/api/v1/stock/{symbol}/financials",
            "stock_news": "/api/v1/stock/{symbol}/news",
            "stock_price": "/api/v1/stock/{symbol}/price",
            "analyze": "/api/v1/analyze (POST)",
            "report": "/api/v1/report (POST)",
            "compare": "/api/v1/compare (POST)"
        },
        "configured": {
            "finnhub": bool(os.environ.get("FINNHUB_API_KEY")),
            "fmp": bool(os.environ.get("FMP_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY"))
        }
    }

@app.get("/health")
async def health():
    """Simple health check"""
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
        
        formatted = f"""[Company Profile]
{profile.get('name', 'N/A')} is a leading entity in the {profile.get('finnhubIndustry', 'N/A')} sector.
Incorporated and publicly traded since {profile.get('ipo', 'N/A')}.
Market Cap: {profile.get('marketCapitalization', 0):.2f}M {profile.get('currency', 'USD')}
Shares Outstanding: {profile.get('shareOutstanding', 0):.2f}M
Exchange: {profile.get('exchange', 'N/A')}
Country: {profile.get('country', 'N/A')}
Website: {profile.get('weburl', 'N/A')}"""
        
        return {
            "symbol": symbol.upper(),
            "profile": formatted,
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
        
        key_metrics = {
            "valuation": {
                "pe_ratio": m.get("peTTM"),
                "forward_pe": m.get("forwardPE"),
                "ps_ratio": m.get("psTTM"),
                "pb_ratio": m.get("pbQuarterly"),
                "ev_ebitda": m.get("evEbitdaTTM"),
                "peg_ratio": m.get("pegTTM"),
            },
            "profitability": {
                "gross_margin": m.get("grossMarginTTM"),
                "operating_margin": m.get("operatingMarginTTM"),
                "net_margin": m.get("netProfitMarginTTM"),
                "roe": m.get("roeTTM"),
                "roi": m.get("roiTTM"),
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
            },
            "performance": {
                "ytd_return": m.get("yearToDatePriceReturnDaily"),
                "52_week_return": m.get("52WeekPriceReturnDaily"),
            }
        }
        
        return {
            "symbol": symbol.upper(),
            "key_metrics": key_metrics,
            "raw_metrics": m,
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
        
        # Format news
        formatted_news = []
        for item in news[:20]:  # Limit to 20 articles
            formatted_news.append({
                "headline": item.get("headline"),
                "summary": item.get("summary", "")[:300],
                "source": item.get("source"),
                "url": item.get("url"),
                "datetime": datetime.fromtimestamp(item.get("datetime", 0)).isoformat()
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
    """Get stock price data using yfinance"""
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
# Analysis Endpoints
# ============================================================================

@app.post("/api/v1/analyze")
async def analyze_stock(request: StockAnalysisRequest):
    """Comprehensive AI-powered stock analysis"""
    try:
        import yfinance as yf
        
        symbol = request.symbol.upper()
        client = get_finnhub_client()
        
        # Get financial data
        metrics = client.company_basic_financials(symbol, 'all')
        m = metrics.get('metric', {}) if metrics else {}
        
        # Get news
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        news = client.company_news(
            symbol,
            _from=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d")
        )
        
        # Get price data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # Calculate score and insights
        score = 0
        insights = []
        
        # Valuation analysis
        pe = m.get("peTTM") or 0
        if 0 < pe < 15:
            score += 3
            insights.append(f"Very attractive P/E ratio: {pe:.1f}x")
        elif 0 < pe < 25:
            score += 2
            insights.append(f"Reasonable P/E ratio: {pe:.1f}x")
        elif pe > 50:
            score -= 1
            insights.append(f"High P/E ratio: {pe:.1f}x - premium valuation")
        
        # PEG ratio
        peg = m.get("pegTTM") or 0
        if 0 < peg < 1:
            score += 2
            insights.append(f"PEG ratio {peg:.2f} suggests undervalued relative to growth")
        elif 1 <= peg < 2:
            score += 1
            insights.append(f"Fair PEG ratio: {peg:.2f}")
        
        # Growth analysis
        rev_growth = m.get("revenueGrowthTTMYoy") or 0
        if rev_growth > 30:
            score += 2
            insights.append(f"Excellent revenue growth: {rev_growth:.1f}% YoY")
        elif rev_growth > 10:
            score += 1
            insights.append(f"Solid revenue growth: {rev_growth:.1f}% YoY")
        elif rev_growth < 0:
            score -= 1
            insights.append(f"Revenue declining: {rev_growth:.1f}% YoY")
        
        eps_growth = m.get("epsGrowthTTMYoy") or 0
        if eps_growth > 30:
            score += 2
            insights.append(f"Strong EPS growth: {eps_growth:.1f}% YoY")
        elif eps_growth > 10:
            score += 1
            insights.append(f"Positive EPS growth: {eps_growth:.1f}% YoY")
        
        # Profitability
        net_margin = m.get("netProfitMarginTTM") or 0
        if net_margin > 20:
            score += 2
            insights.append(f"High profitability: {net_margin:.1f}% net margin")
        elif net_margin > 10:
            score += 1
            insights.append(f"Good profitability: {net_margin:.1f}% net margin")
        
        # 52-week position
        high_52 = m.get("52WeekHigh") or 0
        low_52 = m.get("52WeekLow") or 0
        if high_52 > 0 and low_52 > 0 and current_price > 0:
            range_position = (current_price - low_52) / (high_52 - low_52) * 100
            insights.append(f"Trading at {range_position:.0f}% of 52-week range")
        
        # Determine rating
        if score >= 8:
            rating = "STRONG BUY"
        elif score >= 5:
            rating = "BUY"
        elif score >= 2:
            rating = "HOLD"
        elif score >= 0:
            rating = "UNDERPERFORM"
        else:
            rating = "SELL"
        
        return {
            "symbol": symbol,
            "rating": rating,
            "score": score,
            "current_price": round(current_price, 2),
            "insights": insights,
            "key_metrics": {
                "pe_ratio": pe,
                "peg_ratio": peg,
                "revenue_growth": rev_growth,
                "eps_growth": eps_growth,
                "net_margin": net_margin,
                "beta": m.get("beta"),
                "52_week_high": high_52,
                "52_week_low": low_52,
            },
            "news_articles": len(news) if news else 0,
            "analysis_type": request.analysis_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/report")
async def generate_report(request: ReportRequest):
    """Generate equity research report"""
    try:
        import yfinance as yf
        
        symbol = request.symbol.upper()
        company_name = request.company_name
        client = get_finnhub_client()
        
        # Get data
        metrics = client.company_basic_financials(symbol, 'all')
        m = metrics.get('metric', {}) if metrics else {}
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        report = {
            "title": f"{company_name} ({symbol}) Equity Research Report",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "symbol": symbol,
            "company_name": company_name,
            "current_price": round(current_price, 2),
            
            "summary": {
                "market_cap_millions": m.get("marketCapitalization"),
                "ytd_return": m.get("yearToDatePriceReturnDaily"),
                "52_week_high": m.get("52WeekHigh"),
                "52_week_low": m.get("52WeekLow"),
            },
            
            "valuation": {
                "pe_ratio": m.get("peTTM"),
                "forward_pe": m.get("forwardPE"),
                "ps_ratio": m.get("psTTM"),
                "pb_ratio": m.get("pbQuarterly"),
                "ev_ebitda": m.get("evEbitdaTTM"),
                "peg_ratio": m.get("pegTTM"),
            },
            
            "profitability": {
                "gross_margin": m.get("grossMarginTTM"),
                "operating_margin": m.get("operatingMarginTTM"),
                "net_margin": m.get("netProfitMarginTTM"),
                "roe": m.get("roeTTM"),
                "roi": m.get("roiTTM"),
            },
            
            "growth": {
                "revenue_growth_yoy": m.get("revenueGrowthTTMYoy"),
                "eps_growth_yoy": m.get("epsGrowthTTMYoy"),
                "revenue_growth_5y": m.get("revenueGrowth5Y"),
                "eps_growth_5y": m.get("epsGrowth5Y"),
            },
            
            "risk": {
                "beta": m.get("beta"),
                "current_ratio": m.get("currentRatioQuarterly"),
            }
        }
        
        return {
            "report": report,
            "status": "success",
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
        
        for symbol in request.symbols[:5]:  # Limit to 5
            try:
                sym = symbol.upper()
                metrics = client.company_basic_financials(sym, 'all')
                m = metrics.get('metric', {}) if metrics else {}
                
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="1mo")
                current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
                
                results.append({
                    "symbol": sym,
                    "price": round(current_price, 2),
                    "pe_ratio": m.get("peTTM"),
                    "peg_ratio": m.get("pegTTM"),
                    "revenue_growth": m.get("revenueGrowthTTMYoy"),
                    "net_margin": m.get("netProfitMarginTTM"),
                    "beta": m.get("beta"),
                })
            except Exception as e:
                results.append({
                    "symbol": symbol.upper(),
                    "error": str(e)
                })
        
        # Sort by PE ratio
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
