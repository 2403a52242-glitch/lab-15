import argparse
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#!/usr/bin/env python3
"""
Preprocess sales transaction data.

Usage:
    python task_2.py --input sales_transactions.csv --output preprocessed_transactions.csv
Default column names:
    transaction_date
    transaction_amount
"""


def preprocess(df, date_col="transaction_date", amount_col="transaction_amount"):
        # Create a copy of the DataFrame at the start to avoid chained assignment warnings
        df = df.copy()

        # 1) Convert transaction dates to datetime and drop invalid dates
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        # 2) Convert amounts to numeric and remove non-positive values
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
        df = df[df[amount_col] > 0]

        if df.empty:
            return df

        # 3) Create Month-Year column using loc to avoid warnings
        df.loc[:, "Month-Year"] = df[date_col].dt.strftime("%b-%Y")  # e.g., Jan-2020

        # 4) Normalize transaction_amount using Min-Max scaling
        scaler = MinMaxScaler()
        df.loc[:, amount_col + "_normalized"] = scaler.fit_transform(df[[amount_col]])

        return df

def main():
        parser = argparse.ArgumentParser(description="Preprocess sales transaction CSV.")
        parser.add_argument("--input", "-i", default="sales_transactions.csv",
                                                help="Input CSV file (default: sales_transactions.csv)")
        parser.add_argument("--output", "-o", default="preprocessed_transactions.csv",
                                                help="Output CSV file (default: preprocessed_transactions.csv)")
        parser.add_argument("--date-col", default="transaction_date", help="Name of the date column")
        parser.add_argument("--amount-col", default="transaction_amount", help="Name of the amount column")
        args = parser.parse_args()

        try:
                df = pd.read_csv(args.input)
        except Exception as e:
                sys.exit(f"Failed to read input file '{args.input}': {e}")

        processed = preprocess(df, date_col=args.date_col, amount_col=args.amount_col)

        try:
                processed.to_csv(args.output, index=False)
        except Exception as e:
                sys.exit(f"Failed to write output file '{args.output}': {e}")

        print(f"Preprocessing complete. Rows before: {len(df)}, after: {len(processed)}")
        print(f"Saved preprocessed data to: {args.output}")

if __name__ == "__main__":
        main()