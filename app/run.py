import uvicorn

from app.main import application

if __name__ == "__main__":
    uvicorn.run(application)
