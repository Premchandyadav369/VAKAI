import uvicorn

if __name__ == "__main__":
    print("Starting VAK-AI Voice Detector...")
    print("Documentation available at: http://localhost:8000/docs")
    # Reload=True for dev, remove for prod
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
