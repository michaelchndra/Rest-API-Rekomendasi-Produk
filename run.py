# run.py
import os
import subprocess

# Step 1: Run scrape.py for data scraping
subprocess.run(["python", "scrape.py"])

# Step 2: Run train.py for model training
subprocess.run(["python", "train.py"])

# Step 3: Run app.py to start the Flask server
os.system("python app.py")
