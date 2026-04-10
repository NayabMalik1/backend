import uvicorn
import os
import sys
print(sys.executable)

def main():
    """
    Start FastAPI backend server
    """

    print("\n====================================")
    print(" Android Malware FSL Backend Server ")
    print("====================================\n")

    print("Server starting...")
    print("API URL: http://127.0.0.1:8000")
    print("Docs: http://127.0.0.1:8000/docs\n")

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
