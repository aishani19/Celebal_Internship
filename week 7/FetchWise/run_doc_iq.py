import subprocess
import sys
import os
import time

def run():
    print("========================================")
    print("     DocumentIQ RAG Chatbot Startup     ")
    print("========================================")
    
    # Ensure we are in the root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    chatbot_dir = os.path.join(root_dir, "Chatbot")
    
    if os.path.exists(chatbot_dir):
        os.chdir(chatbot_dir)
    elif os.path.exists(os.path.join(root_dir, "src", "main.py")):
        # We are already in Chatbot dir
        pass
    else:
        print("Error: Could not find 'Chatbot' directory or 'src' source code.")
        sys.exit(1)
        
    print("\nStarting FastAPI Backend...")
    # Use the python executable from the virtual environment if it exists
    python_exe = sys.executable
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        python_exe = venv_python
        print(f"Using virtual environment Python: {python_exe}")
        
    # Running uvicorn via python module
    backend_cmd = [python_exe, "-m", "uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", "8000"]
    backend_process = subprocess.Popen(backend_cmd)
    
    print("Waiting 4 seconds for backend to initialize...")
    time.sleep(4)
    
    print("\nStarting Streamlit Frontend...")
    # Start streamlit via python module to avoid wrapper issues
    frontend_cmd = [python_exe, "-m", "streamlit", "run", "src/streamlit_app.py", "--server.runOnSave", "false"]
    frontend_process = subprocess.Popen(frontend_cmd)
    
    print("\nDocumentIQ is now running!")
    print("- Backend API: http://127.0.0.1:8000")
    print("- Frontend UI: http://127.0.0.1:8501")
    print("\nPress Ctrl+C to stop both services gracefully.")
    
    # Wait for the user to press Key to stop
    try:
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down DocumentIQ...")
    finally:
        print("Terminating frontend and backend gracefuly...")
        if frontend_process.poll() is None:
            frontend_process.terminate()
        if backend_process.poll() is None:
            backend_process.terminate()
        backend_process.wait()
        print("Done.")

if __name__ == "__main__":
    run()
