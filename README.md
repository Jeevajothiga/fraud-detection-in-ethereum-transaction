# fraud-detection-in-ethereum-transaction
# Ethereum Transaction Fraud Detection System

## Overview

This project detects fraud in Ethereum transactions by fetching wallet data from the Etherscan API, processing it, identifying suspicious patterns, and training a Random Forest model.

## Features

- Fetch Ethereum transactions using the Etherscan API
- Process and clean transaction data
- Detect various fraud types including:
  - Money Laundering
  - Rug Pulls
  - Flash Loan Attacks
  - Honeypot Scams
  - Wash Trading
  - And more
- Train and evaluate a Random Forest classifier on labeled data
- Display transaction timestamps in UTC and IST for easy interpretation

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ethereum-fraud-detection.git
   cd ethereum-fraud-detection
Install required Python packages:

  ```bash
  pip install -r requirements.txt
Usage
Obtain your Etherscan API key and update the ETHERSCAN_API_KEY variable in the script.

run the fraud detection script:

  ```bash
  python fraud_detection.py
To analyze a different wallet, update the wallet_address variable in the script.

Code Breakdown
fetch_transactions(wallet_address): Fetches Ethereum transaction data using the Etherscan API.

process_transactions(df): Cleans and preprocesses transaction data (converts timestamps, values, etc.).

identify_fraud(df): Detects fraud by analyzing transactions and assigning risk scores and fraud types.

train_fraud_model(): Trains a Random Forest classifier on labeled transaction data for fraud prediction.

Contributing
Feel free to fork the repo and submit pull requests for improvements or new features.
