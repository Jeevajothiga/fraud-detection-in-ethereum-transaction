# Ethereum Transaction Fraud Detection System

## Overview

This project aims to detect fraudulent Ethereum transactions by fetching wallet data from the Etherscan API, processing it, identifying suspicious patterns, and training a Random Forest model for fraud prediction.

## Features

- Fetch Ethereum transactions using the **Etherscan API**
- Process and clean transaction data
- Detect various fraud types, including:
  - **Money Laundering**
  - **Rug Pulls**
  - **Flash Loan Attacks**
  - **Honeypot Scams**
  - **Wash Trading**  
  - ...and more!
- Train and evaluate a **Random Forest classifier** on labeled transaction data
- Display transaction timestamps in **UTC and IST** for easy interpretation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ethereum-fraud-detection.git
   cd ethereum-fraud-detection
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Obtain your **Etherscan API key** and update the `ETHERSCAN_API_KEY` variable in the script.
2. Run the fraud detection script:
   ```bash
   python fraud_detection.py
   ```
3. To analyze a different wallet, update the `wallet_address` variable in the script.

## Code Breakdown

- `fetch_transactions(wallet_address)`: Fetches Ethereum transaction data using the Etherscan API.
- `process_transactions(df)`: Cleans and preprocesses transaction data (e.g., timestamp conversions, value formatting).
- `identify_fraud(df)`: Analyzes transactions, assigns risk scores, and categorizes fraud types.
- `train_fraud_model()`: Trains a **Random Forest classifier** on labeled transaction data for fraud prediction.

## Contributing

Contributions are welcome! Fork the repo and submit pull requests for improvements or new features.

