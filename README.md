# Repeated Median Velocity Trading Strategy

The **Repeated-Median-Velocity-Trading-Strategy** is a quantitative trading strategy that utilizes the Alpaca Stock API to fetch market data and implements a repeated median velocity trading indicator. This project employs walk-forward optimization to identify optimal system parameters for intraday trading. The strategy is inspired by the research paper: [The Robust Repeated Median Velocity System](https://meyersanalytics.com/publications2/es5rmed2.pdf) by Dennis Meyers.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)

## Introduction

The Repeated-Median Velocity is designed to capture trends efficiently while minimizing effects from outlier data by using a robust statistical method known as the repeated median regression (Siegel Slope).

This project provides a systematic approach to developing, backtesting, and optimizing trading strategies using vectorbt and the Alpaca API with Python.

## Features

- **Repeated Median Velocity Indicator**: Implements the repeated median method
- **Walk-Forward Optimization**: Utilizes vectorbt walk-forward optimization techniques to determine the best strategy parameters for each trading week.
- **Alpaca Stock API Integration**: Fetches historical data from the Alpaca API (can be extended for real-time data).
- **Intraday Trading Focus**: Constructed for executing trades within a single trading day.
- **Robust Backtesting**: Uses vecortbt's tools to test strategy performance against historical data and highlight trading metrics.


## Setup

To get started with the Repeated-Median-Velocity-Trading-Strategy, follow these steps:

### Prerequisites

- Python with libraries found in `requirements.txt`
- Alpaca account with API key and secret

### Clone the Repository

```bash
git clone https://github.com/0zean/Repeated-Median-Velocity-Trading-Strategy.git
cd Repeated-Median-Velocity-Trading-Strategy
```

### Install dependencies

```bash
pip install -r requiremnts.txt
```
