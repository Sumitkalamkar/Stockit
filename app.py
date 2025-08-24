from fastapi import FastAPI
from routes.stock_routes import router as stock_router  # Import the router

app = FastAPI(title="Stock Predictor API")

# Include the stock router with a prefix
app.include_router(stock_router, prefix="/stock")

@app.get("/")
def root():
    return {"message": "Welcome to the Stock Predictor API!"}

# Test route to ensure server is running
@app.get("/test")
def test():
    return {"status": "API is running"}

# Optional: catch-all route to handle extra trailing slashes or newlines
@app.middleware("http")
async def strip_whitespace(request, call_next):
    request.scope["path"] = request.scope["path"].rstrip("/\n")
    response = await call_next(request)
    return response
