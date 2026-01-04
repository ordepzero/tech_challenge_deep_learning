from fastapi import FastAPI
from src.api import routes_model, routes_data, routes_utils 

app = FastAPI()

app.include_router(routes_model.router)
app.include_router(routes_data.router)
app.include_router(routes_utils.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}