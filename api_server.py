"""
FinRobot API Server - Railway Optimized
AI-powered Equity Research and Financial Analysis
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

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
# Startup Configuration
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize API keys on startup"""
    # Check for environment variables (Railway sets these)
    required_keys = ["FINNHUB_API_KEY"]
    optional_keys = ["FMP_API_KEY", "SEC_API_KEY", "OPENAI_API_KEY"]
    
    print("=" * 50)
    print("FinRobot API Starting...")
    print("=" * 50)
    
    for key in required_keys:
        if os.environ.get(key):
            print(f"✓ {key} configured")
        else:
            print(f"✗ {key} MISSING (required)")
    
    for key in optional_keys:
        if os.environ.get(key):
            print(f"✓ {key} configured")
        else:
            print(f"○ {key} not set (optional)")
    
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
    symbols: list[str]

# ============================================================================
# Health & Info Endpoints
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
        "data_sources": {
            "finnhub": bool(os.environ.get("FINNHUB_API_KEY")),
            "fmp": bool(os.environ.get("FMP_API_KEY")),
            "yahoo_finance": True
        }
    }

@app.get("/health")
async def health():
    """Simple health check for Railway"""
    return {"status": "ok"}

# ============================================================================
# Stock Data Endpoints
# ============================================================================

@app.get("/api/v1/stock/{symbol}/profile")
async def get_stock_profile(symbol: str):
    """Get company profile"""
    try:
        from finrobot.data_source.finnhub_utils import FinnHubUtils
        profile = FinnHubUtils.get_company_profile(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "profile": profile,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stock/{symbol}/financials")
async def get_stock_financials(symbol: str):
    """Get detailed financial metrics"""
    try:
        from finrobot.data_source.finnhub_utils import FinnHubUtils
        financials = FinnHubUtils.get_basic_financials(symbol.upper())
        
        # Parse if string
        if isinstance(financials, str):
            try:
                financials = json.loads(financials)
            except:
                pass
        
        # Extract key metrics for easier consumption
        key_metrics = {}
        if isinstance(financials, dict):
            key_metrics = {
                "valuation": {
                    "pe_ratio": financials.get("peTTM"),
                    "forward_pe": financials.get("forwardPE"),
                    "ps_ratio": financials.get("psTTM"),
                    "pb_ratio": financials.get("pbQuarterly"),
                    "ev_ebitda": financials.get("evEbitdaTTM"),
                    "peg_ratio": financials.get("pegTTM"),
                },
                "profitability": {
                    "gross_margin": financials.get("grossMarginTTM"),
                    "operating_margin": financials.get("operatingMarginTTM"),
                    "net_margin": financials.get("netProfitMarginTTM"),
                    "roe": financials.get("roeTTM"),
                    "roi": financials.get("roiTTM"),
                },
                "growth": {
                    "revenue_growth_yoy": financials.get("revenueGrowthTTMYoy"),
                    "eps_growth_yoy": financials.get("epsGrowthTTMYoy"),
                    "revenue_growth_5y": financials.get("revenueGrowth5Y"),
                    "eps_growth_5y": financials.get("epsGrowth5Y"),
                },
                "risk": {
                    "beta": financials.get("beta"),
                    "52_week_high": financials.get("52WeekHigh"),
                    "52_week_low": financials.get("52WeekLow"),
                }
            }
        
        return {
            "symbol": symbol.upper(),
            "key_metrics": key_metrics,
            "raw_data": financials,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stock/{symbol}/news")
async def get_stock_news(symbol: str, days: int = 7):
    """Get recent company news"""
    try:
        from finrobot.data_source.finnhub_utils import FinnHubUtils
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        news = FinnHubUtils.get_company_news(symbol.upper(), start_date, end_date)
        
        return {
            "symbol": symbol.upper(),
            "news": news,
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
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        current = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[0])
        change_pct = ((current - prev) / prev) * 100
        
        # Get basic info
        info = ticker.info
        
        return {
            "symbol": symbol.upper(),
            "current_price": round(current, 2),
            "period_start_price": round(prev, 2),
            "change_percent": round(change_pct, 2),
            "period_high": round(float(hist['High'].max()), 2),
            "period_low": round(float(hist['Low'].min()), 2),
            "avg_volume": int(hist['Volume'].mean()),
            "company_name": info.get("longName", symbol.upper()),
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
    """
    Comprehensive AI-powered stock analysis
    """
    try:
        from finrobot.data_source.finnhub_utils import FinnHubUtils
        import yfinance as yf
        
        symbol = request.symbol.upper()
        
        # Gather all data
        profile = FinnHubUtils.get_company_profile(symbol)
        financials_raw = FinnHubUtils.get_basic_financials(symbol)
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        news = FinnHubUtils.get_company_news(symbol, start_date, end_date)
        
        # Get price data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # Parse financials
        financials = json.loads(financials_raw) if isinstance(financials_raw, str) else financials_raw
        
        # Calculate simple score
        score = 0
        insights = []
        
        # Valuation analysis
        pe = financials.get("peTTM", 0) or 0
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
        peg = financials.get("pegTTM", 0) or 0
        if 0 < peg < 1:
            score += 2
            insights.append(f"PEG ratio {peg:.2f} suggests undervalued relative to growth")
        elif 1 <= peg < 2:
            score += 1
            insights.append(f"Fair PEG ratio: {peg:.2f}")
        
        # Growth analysis
        rev_growth = financials.get("revenueGrowthTTMYoy", 0) or 0
        if rev_growth > 30:
            score += 2
            insights.append(f"Excellent revenue growth: {rev_growth:.1f}% YoY")
        elif rev_growth > 10:
            score += 1
            insights.append(f"Solid revenue growth: {rev_growth:.1f}% YoY")
        elif rev_growth < 0:
            score -= 1
            insights.append(f"Revenue declining: {rev_growth:.1f}% YoY")
        
        eps_growth = financials.get("epsGrowthTTMYoy", 0) or 0
        if eps_growth > 30:
            score += 2
            insights.append(f"Strong EPS growth: {eps_growth:.1f}% YoY")
        elif eps_growth > 10:
            score += 1
            insights.append(f"Positive EPS growth: {eps_growth:.1f}% YoY")
        
        # Profitability
        net_margin = financials.get("netProfitMarginTTM", 0) or 0
        if net_margin > 20:
            score += 2
            insights.append(f"High profitability: {net_margin:.1f}% net margin")
        elif net_margin > 10:
            score += 1
            insights.append(f"Good profitability: {net_margin:.1f}% net margin")
        
        # ROE
        roe = financials.get("roeTTM", 0) or 0
        if isinstance(roe, (int, float)) and roe > 0.2:
            score += 1
            insights.append(f"Strong ROE: {roe*100:.1f}%")
        
        # Price momentum
        if not hist.empty and len(hist) > 20:
            month_ago = float(hist['Close'].iloc[-20])
            momentum = ((current_price - month_ago) / month_ago) * 100
            if momentum > 10:
                insights.append(f"Positive momentum: +{momentum:.1f}% (1mo)")
            elif momentum < -10:
                insights.append(f"Negative momentum: {momentum:.1f}% (1mo)")
        
        # 52-week position
        high_52 = financials.get("52WeekHigh", 0) or 0
        low_52 = financials.get("52WeekLow", 0) or 0
        if high_52 > 0 and low_52 > 0:
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
        
        # News sentiment summary
        news_count = len(news) if isinstance(news, list) else 0
        
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
                "roe": roe,
                "beta": financials.get("beta"),
                "52_week_high": high_52,
                "52_week_low": low_52,
            },
            "news_articles": news_count,
            "analysis_type": request.analysis_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/report")
async def generate_report(request: ReportRequest):
    """Generate equity research report"""
    try:
        from finrobot.data_source.finnhub_utils import FinnHubUtils
        import yfinance as yf
        
        symbol = request.symbol.upper()
        company_name = request.company_name
        
        # Get data
        financials_raw = FinnHubUtils.get_basic_financials(symbol)
        financials = json.loads(financials_raw) if isinstance(financials_raw, str) else financials_raw
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # Calculate YTD return
        ytd_return = financials.get("yearToDatePriceReturnDaily", 0)
        
        report = {
            "title": f"{company_name} ({symbol}) Equity Research Report",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "symbol": symbol,
            "company_name": company_name,
            "current_price": round(current_price, 2),
            
            "summary": {
                "market_cap_millions": financials.get("marketCapitalization"),
                "ytd_return": ytd_return,
                "52_week_high": financials.get("52WeekHigh"),
                "52_week_low": financials.get("52WeekLow"),
            },
            
            "valuation": {
                "pe_ratio": financials.get("peTTM"),
                "forward_pe": financials.get("forwardPE"),
                "ps_ratio": financials.get("psTTM"),
                "pb_ratio": financials.get("pbQuarterly"),
                "ev_ebitda": financials.get("evEbitdaTTM"),
                "peg_ratio": financials.get("pegTTM"),
            },
            
            "profitability": {
                "gross_margin": financials.get("grossMarginTTM"),
                "operating_margin": financials.get("operatingMarginTTM"),
                "net_margin": financials.get("netProfitMarginTTM"),
                "roe": financials.get("roeTTM"),
                "roi": financials.get("roiTTM"),
            },
            
            "growth": {
                "revenue_growth_yoy": financials.get("revenueGrowthTTMYoy"),
                "eps_growth_yoy": financials.get("epsGrowthTTMYoy"),
                "revenue_growth_5y": financials.get("revenueGrowth5Y"),
                "eps_growth_5y": financials.get("epsGrowth5Y"),
            },
            
            "risk": {
                "beta": financials.get("beta"),
                "current_ratio": financials.get("currentRatioQuarterly"),
                "debt_to_equity": financials.get("totalDebt/totalEquityQuarterly"),
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
        results = []
        
        for symbol in request.symbols[:5]:  # Limit to 5 stocks
            try:
                # Use the analyze endpoint logic
                analysis_request = StockAnalysisRequest(symbol=symbol)
                # Inline analysis to avoid circular call
                from finrobot.data_source.finnhub_utils import FinnHubUtils
                import yfinance as yf
                
                sym = symbol.upper()
                financials_raw = FinnHubUtils.get_basic_financials(sym)
                financials = json.loads(financials_raw) if isinstance(financials_raw, str) else financials_raw
                
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="1mo")
                current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
                
                results.append({
                    "symbol": sym,
                    "price": round(current_price, 2),
                    "pe_ratio": financials.get("peTTM"),
                    "peg_ratio": financials.get("pegTTM"),
                    "revenue_growth": financials.get("revenueGrowthTTMYoy"),
                    "net_margin": financials.get("netProfitMarginTTM"),
                    "beta": financials.get("beta"),
                })
            except Exception as e:
                results.append({
                    "symbol": symbol.upper(),
                    "error": str(e)
                })
        
        # Sort by PE ratio (lower is better, but filter out None/0)
        valid_results = [r for r in results if r.get("pe_ratio") and r.get("pe_ratio") > 0]
        valid_results.sort(key=lambda x: x.get("pe_ratio", 999))
        
        return {
            "comparison": results,
            "best_value": valid_results[0]["symbol"] if valid_results else None,
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
