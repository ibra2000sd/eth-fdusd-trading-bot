# ETH/FDUSD Advanced Trading Bot - Binance API Setup Guide

## Table of Contents

1. [Introduction to Binance API](#introduction-to-binance-api)
2. [Creating a Binance Account](#creating-a-binance-account)
3. [Securing Your Account](#securing-your-account)
4. [Generating API Keys](#generating-api-keys)
5. [Configuring API Permissions](#configuring-api-permissions)
6. [IP Whitelisting for Security](#ip-whitelisting-for-security)
7. [Testnet API Configuration](#testnet-api-configuration)
8. [Integrating API Keys with the Bot](#integrating-api-keys-with-the-bot)
9. [API Key Management Best Practices](#api-key-management-best-practices)
10. [Troubleshooting API Issues](#troubleshooting-api-issues)

---

## Introduction to Binance API

The Binance API (Application Programming Interface) provides a powerful way to interact with the Binance exchange programmatically. It allows the ETH/FDUSD Advanced Trading Bot to perform essential functions such as:

- **Real-time Market Data**: Accessing live price feeds, order book data, and trade history.
- **Order Execution**: Placing, canceling, and managing market, limit, and stop-loss orders.
- **Account Management**: Retrieving account balances, position information, and trade history.
- **Historical Data**: Downloading historical kline data for backtesting and analysis.

The trading bot uses the Binance API to execute its proprietary mathematical models and trading strategies. Proper API setup is critical for the bot's performance, security, and reliability.

### API Key Components

A Binance API key consists of two parts:

- **API Key**: A public identifier for your API access.
- **Secret Key**: A private key used to sign requests, proving your identity.

**Important**: The Secret Key is displayed only once upon creation. Treat it like a password and store it securely. Never share it or commit it to version control.

### API Endpoints

The trading bot interacts with several Binance API endpoints:

- **REST API**: For synchronous operations like placing orders and retrieving account information.
- **WebSocket API**: For real-time data streams, including price tickers, order book updates, and trade feeds.

---

## Creating a Binance Account

If you do not have a Binance account, you will need to create one to use the trading bot.

1. **Visit Binance Website**: Go to [www.binance.com](https://www.binance.com) and click "Register".
2. **Enter Your Details**: Provide your email address or phone number and create a secure password.
3. **Verify Your Account**: Complete the email or phone verification process.
4. **Complete Identity Verification (KYC)**: For live trading, you must complete Binance's identity verification process. This involves providing personal information and a government-issued ID.

---

## Securing Your Account

Before generating API keys, it is crucial to secure your Binance account to protect your funds.

### Two-Factor Authentication (2FA)

Enable two-factor authentication (2FA) for both login and API key creation. This adds an extra layer of security by requiring a second verification step.

**Recommended 2FA Methods:**
- **Google Authenticator**: A time-based one-time password (TOTP) application.
- **YubiKey**: A hardware security key for the highest level of security.

### Anti-Phishing Code

Set up an anti-phishing code in your Binance account settings. This code will be included in all official Binance emails, helping you identify and avoid phishing attempts.

---

## Generating API Keys

Follow these steps to generate API keys for the trading bot.

1. **Log in to Binance**: Access your Binance account.
2. **Navigate to API Management**: Go to your user dashboard and select "API Management".
3. **Create a New API Key**: Enter a descriptive label for your API key (e.g., "ETH-Trading-Bot") and click "Create API".
4. **Complete Security Verification**: Enter your 2FA code and email verification code.
5. **Save Your Keys**: Your API Key and Secret Key will be displayed. **Save them immediately in a secure location**. The Secret Key will not be shown again.

---

## Configuring API Permissions

When creating your API key, you must configure the correct permissions for the trading bot to function properly.

**Required Permissions:**
- **Enable Reading**: Allows the bot to access market data and account information.
- **Enable Spot & Margin Trading**: Allows the bot to execute trades in the spot market.

**Permissions to AVOID:**
- **Enable Withdrawals**: **NEVER** enable withdrawal permissions for your trading bot API key. This is a major security risk.
- **Enable Margin**: Only enable if you intend to use margin trading (not recommended for beginners).
- **Enable Futures**: Only enable if you are using the futures version of the bot.

---

## IP Whitelisting for Security

IP whitelisting is a critical security feature that restricts API access to a specific set of IP addresses. This prevents unauthorized use of your API keys even if they are compromised.

1. **Identify Your Server IP**: Get the public IP address of the server where you will run the trading bot.
   ```bash
   curl ifconfig.me
   ```
2. **Edit API Restrictions**: In the API Management section, click "Edit restrictions" for your API key.
3. **Select "Restrict access to trusted IPs only"**.
4. **Enter Your Server IP**: Add your server's IP address to the whitelist.
5. **Save Changes**: Confirm the changes with your 2FA code.

---

## Testnet API Configuration

Before deploying the trading bot with real funds, it is essential to test it on the Binance testnet. The testnet is a simulated environment that mirrors the live market, allowing you to test your strategy without financial risk.

1. **Visit Binance Testnet**: Go to [testnet.binance.vision](https://testnet.binance.vision).
2. **Create a Testnet Account**: You will need to create a separate account for the testnet.
3. **Generate Testnet API Keys**: Follow the same process as generating live API keys, but on the testnet website.
4. **Fund Your Testnet Account**: The testnet provides free virtual funds for testing.

---

## Integrating API Keys with the Bot

Once you have your API keys, you need to configure the trading bot to use them.

1. **Locate the .env File**: In the root directory of the trading bot, find the `.env` file. If it doesn't exist, copy it from `.env.example`.
   ```bash
   cp .env.example .env
   ```
2. **Edit the .env File**: Open the `.env` file in a text editor.
3. **Enter Your API Keys**:
   ```bash
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET_KEY=your_secret_key_here
   ```
4. **Configure Testnet Mode**:
   - For testnet trading, set `BINANCE_TESTNET=true`.
   - For live trading, set `BINANCE_TESTNET=false`.

**Security Note**: The `.env` file is included in the `.gitignore` file to prevent it from being accidentally committed to version control. Always keep this file secure and never share it.

---

## API Key Management Best Practices

- **Use Separate Keys**: Create separate API keys for different applications and purposes.
- **Regular Rotation**: Rotate your API keys every 30-90 days to minimize the risk of compromise.
- **Principle of Least Privilege**: Only grant the minimum necessary permissions for each API key.
- **Secure Storage**: Store your API keys in a secure, encrypted location like a password manager or a hardware security module (HSM).
- **Monitor Usage**: Regularly review your API key usage in the Binance dashboard for any suspicious activity.

---

## Troubleshooting API Issues

**Error: "Invalid API-key, IP, or permissions for action"**
- **Solution**: Check your API key permissions and IP whitelist settings. Ensure that your server's IP address is correctly whitelisted.

**Error: "Timestamp for this request is outside of the recvWindow"**
- **Solution**: Your server's time is out of sync with Binance's servers. Synchronize your system time using an NTP client.
  ```bash
  sudo ntpdate -s time.nist.gov
  ```

**Error: "Signature for this request is not valid"**
- **Solution**: This usually indicates an incorrect Secret Key. Double-check that you have copied the Secret Key correctly. Remember that it is only displayed once.

**Error: "API key format is invalid"**
- **Solution**: Ensure that you have copied the API Key and Secret Key without any extra spaces or characters.

By following this guide, you can securely and effectively set up your Binance API keys for use with the ETH/FDUSD Advanced Trading Bot. Proper API management is a cornerstone of successful and secure algorithmic trading.

