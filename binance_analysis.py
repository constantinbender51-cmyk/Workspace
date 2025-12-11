import pandas as pd
import numpy as np
import requests
import time
import datetime as dt
import random
import os

# --- Configuration & Fixed Parameters ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '1 Jan, 2018' # Data fetch starts from 2018
ANNUALIZATION_FACTOR = 365 
REGIME_FILTER_SMA = 400 # New Macro-regime filter: SMA 400

# --- Genetic Algorithm Parameters ---
POPULATION_SIZE = 50
GENERATIONS = 20
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 5

# --- Parameter Bounds (Optimization Space) ---
# [Min, Max, Type, Name]
PARAM_BOUNDS = [
    [0.001, 0.100, 'float', 'Center Distance (C)'],
    [1.0, 10.0, 'float', 'Leverage (L)'],
    [0.10, 1.00, 'float', 'Exponent (E)'],
    [0.00, 0.10, 'float', 'Stop Loss (S)'],
    [0.01, 0.10, 'float', 'Re-entry Proximity (P)'],
    [50, 200, 'int', 'SMA Period (M)']
]

# --- 1. Data Fetching Utilities ---

def date_to_milliseconds(date_str):
    """Convert date string to UTC timestamp in milliseconds"""
    return int(dt.datetime.strptime(date_str, '%d %b, %Y').timestamp() * 1000)

def fetch_klines(symbol, interval, start_str):
    """Fetches historical klines data from Binance."""
    print(f"-> Fetching {symbol} {interval} data starting from {start_str}...")
    start_ts = date_to_milliseconds(start_str)
    base_url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    
    while True:
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'limit': 1000
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status() 
            klines = response.json()
            
            if not klines:
                break

            all_data.extend(klines)
            start_ts = klines[-1][0] + 1
            time.sleep(0.5) 
            
            if len(klines) < 1000:
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
        
    print(f"-> Data fetch complete. Total candles: {len(all_data)}")
    
    df = pd.DataFrame(all_data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
        'Taker Buy Quote Asset Volume', 'Ignore'
    ])

    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
    df = df.set_index('Open Time')
    df = df[['Open', 'High', 'Low', 'Close']]
    
    return df.dropna()

# --- 2. Data Preparation and Splitting ---

def prepare_and_split_data(df):
    """Calculates SMA 400 and splits data into Bull and Bear regimes."""
    
    print(f"\n-> Preparing data using SMA {REGIME_FILTER_SMA} filter...")
    
    # 1. Calculate SMA 400 (Filter) and Raw Daily Returns
    df[f'SMA_{REGIME_FILTER_SMA}'] = df['Close'].rolling(window=REGIME_FILTER_SMA).mean()
    
    # 2. Drop initial NaNs for reliable comparison
    df = df.dropna().copy()
    
    # 3. Create Bull and Bear DataFrames
    df_bull_all = df[df['Close'] > df[f'SMA_{REGIME_FILTER_SMA}']].copy()
    df_bear_all = df[df['Close'] <= df[f'SMA_{REGIME_FILTER_SMA}']].copy()

    print(f"   Total days available for optimization: {len(df)}")
    print(f"   Bull Regime Days: {len(df_bull_all)} ({len(df_bull_all)/len(df)*100:.1f}%)")
    print(f"   Bear Regime Days: {len(df_bear_all)} ({len(df_bear_all)/len(df)*100:.1f}%)")

    # 4. Split each regime data into 50% training segments
    
    # Bull Training Data (First 50% of all bull days)
    bull_split_index = len(df_bull_all) // 2
    df_bull_train = df_bull_all.iloc[:bull_split_index].copy()
    
    # Bear Training Data (First 50% of all bear days)
    bear_split_index = len(df_bear_all) // 2
    df_bear_train = df_bear_all.iloc[:bear_split_index].copy()
    
    print(f"   Bull Training Days (50%): {len(df_bull_train)}")
    print(f"   Bear Training Days (50%): {len(df_bear_train)}")
    
    return df_bull_train, df_bear_train

# --- 3. Metric Calculations (Sharpe) ---

def calculate_sharpe_ratio(returns, annualization_factor=ANNUALIZATION_FACTOR, risk_free_rate=0):
    """Calculates the Annualized Sharpe Ratio."""
    if returns.empty or len(returns) <= 1:
        return -np.inf 
    excess_return = returns - risk_free_rate
    mean_excess_return = excess_return.mean()
    std_dev = excess_return.std()
    
    if std_dev == 0:
        return 0.0

    sharpe = (mean_excess_return * annualization_factor) / (std_dev * np.sqrt(annualization_factor))
    return sharpe

# --- 4. Fitness Function (Strategy Run) ---

def run_strategy_for_fitness(df_data, individual):
    """
    Applies the strategy based on the 6 parameters in the individual array.
    Returns the Annualized Sharpe Ratio (Fitness).
    """
    df = df_data.copy()
    
    # Unpack parameters: [C, L, E, S, P, M]
    center, leverage, exponent, sl_percent, reentry_prox, sma_period = individual
    sma_period = int(sma_period) 

    # 1. Calculate Strategy SMA & Raw Daily Returns
    df[f'SMA_{sma_period}'] = df['Close'].rolling(window=sma_period).mean()
    df['Daily_Return_Raw'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Prepare Lagged Indicators ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA'] = df[f'SMA_{sma_period}'].shift(1)
    df['Proximity_to_SMA'] = np.abs((df['Yesterday_Close'] - df['Yesterday_SMA']) / df['Yesterday_SMA'])
    
    # Drop NaNs created by the strategy SMA and shift operations
    df = df.dropna().copy()
    
    if df.empty:
        return -np.inf

    strategy_returns = pd.Series(index=df.index, dtype=float)
    sl_cooldown = False 
    
    # Iterate through the DataFrame for day-by-day logic
    for i in range(len(df)):
        index = df.index[i]
        
        entry_price = df.loc[index, 'Yesterday_Close']
        yesterday_strategy_sma = df.loc[index, 'Yesterday_SMA']
        proximity = df.loc[index, 'Proximity_to_SMA']
        
        # --- 1. Determine Base Position and Direction ---
        direction = np.where(entry_price > yesterday_strategy_sma, 1, -1)
        distance_d = np.abs((entry_price - yesterday_strategy_sma) / yesterday_strategy_sma)
        
        # Multiplier (M) calculation
        distance_scaler = 1.0 / center
        scaled_distance = distance_d * distance_scaler
        epsilon = 1e-10 
        
        denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
        multiplier = np.where(denominator <= 0, 0, 1.0 / (denominator ** exponent))
        position_size_base = direction * multiplier

        # --- 2. Apply Re-entry/Cooldown Filter (State Management) ---
        
        if sl_cooldown:
            if proximity <= reentry_prox:
                sl_cooldown = False 
                
        if sl_cooldown:
            position_size_base = 0.0
            daily_return = 0.0
            
        else:
            # --- 3. Stop Loss Logic ---
            
            current_low = df.loc[index, 'Low']
            current_high = df.loc[index, 'High']
            raw_return = df.loc[index, 'Daily_Return_Raw']
            
            stop_price = np.where(
                direction == 1,
                entry_price * (1 - sl_percent),
                entry_price * (1 + sl_percent)
            )
            
            if sl_percent > 0.0 and (
                (direction == 1 and current_low <= stop_price) or 
                (direction == -1 and current_high >= stop_price)
            ):
                sl_return = np.log(stop_price / entry_price)
                daily_return = sl_return
                sl_cooldown = True

            else:
                daily_return = raw_return

        # --- 4. Final Strategy Return ---
        strategy_return = daily_return * position_size_base * leverage
        strategy_returns[index] = strategy_return
        
    return calculate_sharpe_ratio(strategy_returns.dropna())

# --- 5. Genetic Algorithm Core Functions ---

def initialize_population():
    """Generates a random initial population within parameter bounds."""
    population = []
    for _ in range(POPULATION_SIZE):
        individual = []
        for low, high, dtype, _ in PARAM_BOUNDS:
            if dtype == 'int':
                gene = random.randint(int(low), int(high))
            else:
                gene = random.uniform(low, high)
            individual.append(gene)
        population.append(individual)
    return population

def select_parents(population, fitnesses):
    """Tournament Selection to choose the fittest parents."""
    parents = []
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    parents.append(sorted_population[0][0])
    
    while len(parents) < POPULATION_SIZE:
        tournament = random.sample(sorted_population, TOURNAMENT_SIZE)
        best_in_tournament = max(tournament, key=lambda x: x[1])
        parents.append(best_in_tournament[0])
        
    return parents

def crossover(parent1, parent2):
    """Single-point crossover."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length.")
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual):
    """Gaussian mutation constrained by parameter bounds."""
    for i, gene in enumerate(individual):
        if random.random() < MUTATION_RATE:
            low, high, dtype, _ = PARAM_BOUNDS[i]
            nudge = random.gauss(0, (high - low) * 0.1) 
            mutated_gene = max(low, min(high, gene + nudge))
            
            if dtype == 'int':
                mutated_gene = int(round(mutated_gene))
            
            individual[i] = mutated_gene
    return individual

def run_optimization_pass(df_train, regime_name):
    """Executes a single GA optimization run."""
    print("\n" + "=" * 80)
    print(f"STARTING GENETIC ALGORITHM OPTIMIZATION: {regime_name} REGIME")
    print(f"Training on: {regime_name} Data (First 50% of observations)")
    print(f"Target Metric: Annualized Sharpe Ratio | Pop Size: {POPULATION_SIZE} | Gens: {GENERATIONS}")
    print("=" * 80)
    
    population = initialize_population()
    best_sharpe_overall = -np.inf
    best_individual = None
    
    for generation in range(GENERATIONS):
        fitnesses = [run_strategy_for_fitness(df_train, individual) for individual in population]
        
        current_best_index = np.argmax(fitnesses)
        current_best_sharpe = fitnesses[current_best_index]
        
        if current_best_sharpe > best_sharpe_overall:
            best_sharpe_overall = current_best_sharpe
            best_individual = population[current_best_index]
        
        parents = select_parents(population, fitnesses)
        
        next_population = []
        for j in range(0, POPULATION_SIZE, 2):
            p1 = parents[j]
            p2 = parents[j+1] if j+1 < POPULATION_SIZE else parents[j]
            
            child1, child2 = crossover(p1, p2)
            
            next_population.append(mutate(child1))
            if len(next_population) < POPULATION_SIZE:
                 next_population.append(mutate(child2))
                 
        population = next_population
        
        print(f"Gen {generation+1}/{GENERATIONS}: Max Sharpe = {current_best_sharpe:.4f} (Overall Best: {best_sharpe_overall:.4f})")

    # Final results display
    print("\n" + "=" * 80)
    print(f"OPTIMIZATION COMPLETE: {regime_name} REGIME")
    print(f"Overall Max Annualized Sharpe Ratio (In-Sample): {best_sharpe_overall:.4f}")
    
    results = {}
    if best_individual:
        for i, (_, _, dtype, name) in enumerate(PARAM_BOUNDS):
            value = best_individual[i]
            if dtype == 'int':
                value = int(value)
            results[name] = value

        print("\nOptimal Parameters Found:")
        for name, value in results.items():
            print(f"  {name}: {value}")
    print("=" * 80)
    
    return results

# --- Main Execution ---

if __name__ == '__main__':
    # 1. Fetch all historical data
    df_data = fetch_klines(SYMBOL, INTERVAL, START_DATE)
    
    if df_data.empty:
        print("Error: Could not retrieve data. Exiting.")
    else:
        # 2. Prepare and split data into training sets
        df_bull_train, df_bear_train = prepare_and_split_data(df_data)
        
        # --- 3. Run Optimization Pass 1: Bull Parameters ---
        if not df_bull_train.empty:
            optimal_bull_params = run_optimization_pass(df_bull_train, "BULL")
        else:
            print("\nSkipping Bull Optimization: Insufficient Bull Regime data for training.")
            optimal_bull_params = None
            
        # --- 4. Run Optimization Pass 2: Bear Parameters ---
        if not df_bear_train.empty:
            optimal_bear_params = run_optimization_pass(df_bear_train, "BEAR")
        else:
            print("\nSkipping Bear Optimization: Insufficient Bear Regime data for training.")
            optimal_bear_params = None

        print("\nOptimization Complete. Please use the following two sets of parameters for your final regime-switching strategy:\n")
        
        if optimal_bull_params:
            print("--- OPTIMAL BULL PARAMETERS ---")
            for name, value in optimal_bull_params.items():
                print(f"  {name}: {value}")

        if optimal_bear_params:
            print("\n--- OPTIMAL BEAR PARAMETERS ---")
            for name, value in optimal_bear_params.items():
                print(f"  {name}: {value}")
        
        if not optimal_bull_params and not optimal_bear_params:
             print("No optimal parameters were found.")
