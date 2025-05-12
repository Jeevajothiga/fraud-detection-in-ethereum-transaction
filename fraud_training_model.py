import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from datetime import datetime
from pytz import timezone
import pytz

ETHERSCAN_API_KEY = "KHRITRM3BF16MHEQRK8MI7TD3DSR5GNBHK"

def fetch_transactions(wallet_address):
    print(f"Fetching transactions for wallet: {wallet_address}")
    all_txns = []
    start_block = 0
    max_txns = 10000
    while True:
        url = (
            f"https://api.etherscan.io/api"
            f"?module=account&action=txlist"
            f"&address={wallet_address}"
            f"&startblock={start_block}"
            f"&endblock=99999999"
            f"&sort=asc"
            f"&apikey={ETHERSCAN_API_KEY}"
        )

        response = requests.get(url)
        data = response.json()

        if data["status"] != "1" or "result" not in data:
            print("Etherscan error or no more transactions:", data.get("message", "Unknown"))
            break

        batch = data["result"]
        if not batch:
            break

        all_txns.extend(batch)
        if len(batch) < max_txns:
            break  # No more pages

        last_block = int(batch[-1]["blockNumber"])
        start_block = last_block + 1  # Move to next block

    print(f"Total transactions fetched: {len(all_txns)}")
    return pd.DataFrame(all_txns)


def process_transactions(df):
    df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s', errors='coerce').dt.tz_localize(None)
    df = df[df['timeStamp'].notna()].copy()

    max_valid_date = pd.Timestamp.utcnow() + pd.DateOffset(years=5)
    max_valid_date = pd.to_datetime(max_valid_date).tz_localize(None)
    df = df[df['timeStamp'] <= max_valid_date].copy()

    # Convert value from Wei to Ether
    df['value'] = df['value'].apply(lambda x: int(x) / 1e18)

    df['transaction_count'] = df.groupby('from')['hash'].transform('count')
    df['is_large_transaction'] = df['value'] > df['value'].quantile(0.90)
    df['time_diff'] = df['timeStamp'].diff().dt.total_seconds().fillna(0)
    df['same_address_tx'] = df['from'] == df['to']
    df['is_fraud'] = 0

    # Display last and current time
    if not df.empty:
        last_tx_time_utc = df['timeStamp'].max().replace(tzinfo=pytz.utc)
        current_time_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        ist = timezone("Asia/Kolkata")
        last_tx_time_ist = last_tx_time_utc.astimezone(ist)
        current_time_ist = current_time_utc.astimezone(ist)

        print("\n🕒 Last Transaction Time:")
        print(f"   • UTC : {last_tx_time_utc.strftime('%d-%m-%Y %I:%M %p')} UTC")
        print(f"   • IST : {last_tx_time_ist.strftime('%d-%m-%Y %I:%M %p')} IST")

        print("🕒 Current Time:")
        print(f"   • UTC : {current_time_utc.strftime('%d-%m-%Y %I:%M %p')} UTC")
        print(f"   • IST : {current_time_ist.strftime('%d-%m-%Y %I:%M %p')} IST\n")

    else:
        print("No valid transactions to process.")

    return df

def is_flash_loan_attack(row):
    return row.get('gasUsed', 0) != 0 and row.get('value', 0) > 300

def is_rug_pull(row):
    return "liquidity" in str(row.get('functionName', '')).lower()

def is_honeypot(row):
    return row['from'] == row.get('contractAddress') and row.get('value', 0) > 0

def is_address_poisoning(row):
    return row.get('value', 0) == 0 and str(row.get('to', '')).startswith("0x000")

def is_fake_airdrop(row):
    return row.get('value', 0) == 0 and "airdrop" in str(row.get('functionName', '')).lower()

def is_wash_trading(row):
    return row.get('from') == row.get('to') and row.get('value', 0) > 0

def identify_fraud(df):
    df['is_fraud'] = 0
    df['fraud_type'] = ''
    df['reason'] = ''
    df['risk_score'] = 0.0

    for index, row in df.iterrows():
        score = 0
        reasons = []
        fraud_type = "Legitimate"

        if row['from'] == row['to']:
            fraud_type = "Cyclic Transaction"
            score += 4
            reasons.append("Sender and receiver are the same")

        elif row['value'] > 100 and row.get('pagerank', 0) > 0.005:
            fraud_type = "Money Laundering"
            score += 4
            reasons.append("Large value transfer and high PageRank")

        elif row.get('out_degree', 0) > 15 and row.get('clustering', 0) > 0.5:
            fraud_type = "Ponzi Scheme"
            score += 4
            reasons.append("High out-degree and clustering")

        elif row.get('pagerank', 0) > 0.008 and row.get('out_degree', 0) > 20:
            fraud_type = "Pump & Dump"
            score += 4
            reasons.append("High PageRank and out-degree")

        elif row['value'] > 500:
            fraud_type = "High-Value Scam"
            score += 4
            reasons.append("Extremely large value transfer (> 500 ETH)")

        elif is_flash_loan_attack(row):
            fraud_type = "Flash Loan Attack"
            score += 4
            reasons.append("Large flash loan followed by suspicious gas usage")

        elif is_rug_pull(row):
            fraud_type = "Rug Pull"
            score += 4
            reasons.append("Liquidity function called before large outflow")

        elif is_honeypot(row):
            fraud_type = "Honeypot Scam"
            score += 4
            reasons.append("Only accepting deposits")

        elif is_address_poisoning(row):
            fraud_type = "Address Poisoning"
            score += 4
            reasons.append("Zero value transaction to lookalike address")

        elif is_fake_airdrop(row):
            fraud_type = "Fake Airdrop Scam"
            score += 4
            reasons.append("Zero-value airdrop to multiple wallets")

        elif is_wash_trading(row):
            fraud_type = "Wash Trading"
            score += 4
            reasons.append("Sender and receiver are same with repeated transactions")

        else:
            fraud_type = "Legitimate"
            reasons.append("No suspicious pattern detected")

        df.at[index, 'risk_score'] = score
        df.at[index, 'is_fraud'] = 1 if score >= 4 else 0
        df.at[index, 'fraud_type'] = fraud_type
        df.at[index, 'reason'] = "; ".join(reasons)

    return df



def train_fraud_model():
    df = fetch_transactions("0x742d35Cc6634C0532925a3b844Bc454e4438f44e")  # Training wallet
    if df.empty:
        print("No transactions found.")
        return None

    df = process_transactions(df)
    df = identify_fraud(df)

    fraud_cases = df[df['is_fraud'] == 1]
    if len(fraud_cases) == 0:
        print("No fraud cases found in training data.")
        return None

    normal_cases_pool = df[df['is_fraud'] == 0]
    if len(normal_cases_pool) == 0:
        print("No normal cases found in training data.")
        return None

    normal_cases = normal_cases_pool.sample(min(len(fraud_cases), len(normal_cases_pool)), replace=True)
    balanced_df = pd.concat([fraud_cases, normal_cases])

    print("\n✅ Label Distribution Before Training:")
    print(balanced_df['is_fraud'].value_counts())

    features = balanced_df[['transaction_count', 'is_large_transaction', 'time_diff', 'same_address_tx']].copy()
    features['is_large_transaction'] = features['is_large_transaction'].astype(int)
    features['same_address_tx'] = features['same_address_tx'].astype(int)

    labels = balanced_df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))

    return model, scaler


if __name__ == "__main__":
    print("Starting fraud detection system...")

    # Train the fraud detection model
    model, scaler = train_fraud_model()
    if model:
        print("\n💡 Fraud detection model trained successfully.")

    wallet_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"  # Replace with the actual wallet address
    transactions = fetch_transactions(wallet_address)
    transactions = process_transactions(transactions)
    transactions = identify_fraud(transactions)
    print("\nFraud Detection Complete:")
    print(transactions[['timeStamp', 'from', 'to', 'value', 'fraud_type', 'reason', 'risk_score']].head())
