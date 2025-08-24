from fastapi import FastAPI
from routes.stock_routes import router as stock_router  # Import stock routes

app = FastAPI(title="Stock Predictor API")

# Include stock routes under /stock
app.include_router(stock_router, prefix="/stock", tags=["Stock"])

@app.get("/")
def root():
    return {"message": "Welcome to the Stock Predictor API!"}

@app.get("/test")
def test():
    return {"status": "API is running"}

# Middleware to strip trailing slash/newline
@app.middleware("http")
async def strip_whitespace(request, call_next):
    request.scope["path"] = request.scope["path"].rstrip("/\n")
    response = await call_next(request)
    return response
