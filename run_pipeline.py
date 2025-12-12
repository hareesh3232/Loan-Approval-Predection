import os
import sys
import subprocess

def run_script(script_name):
    print(f"==================================================")
    print(f"Running {script_name}...")
    print(f"==================================================")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    if result.returncode != 0:
        print(f"Error running {script_name}!")
        sys.exit(result.returncode)
    print("\n")

def main():
    # Define the order of execution
    scripts = [
        os.path.join('src', 'preprocess.py'),
        os.path.join('src', 'train_model.py'),
        os.path.join('src', 'evaluate.py')
    ]

    print("Starting Loan Approval Prediction Pipeline...\n")
    
    for script in scripts:
        script_path = os.path.join('loan_approval_project', script)
        # Check if file exists relative to current dir, or inside project folder
        if not os.path.exists(script_path):
            # Try without project folder prefix if running from inside
            script_path = script
            
        run_script(script_path)

    print("==================================================")
    print("Pipeline Completed Successfully!")
    print("==================================================")
    print("\nTo launch the User Interface, run:")
    print("streamlit run loan_approval_project/app/streamlit_app.py")

if __name__ == "__main__":
    main()
