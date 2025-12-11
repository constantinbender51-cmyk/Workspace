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
ANNUALIZATION_FACTOR = 365 # Used for annualizing Sharpe Ratio for daily data

# --- Genetic Algorithm Parameters ---
POPULATION_SIZE = 50
GENERATIONS = 20
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 5

# --- Optimization Period (BULL Market Window) ---
OPT_START_DATE = '2024-01-01'
OPT_END_DATE = '2025-12-11' # Optimization runs up to the present data

# --- Bear Market Optimized Parameters (Saved) ---
# Center Distance (C): 0.0684588
# Leverage (L): 6.14409
# Exponent (E): 0.84170
# Stop Loss (S): 0.03216
# Re-entry Proximity (P): 0.02910
# SMA Period (M): 157

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

# --- 2. Metric Calculations ---

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

# --- 3. Fitness Function (Strategy Run) ---

def run_strategy_for_fitness(df_data, individual):
    """
    Applies the strategy based on the 6 parameters in the individual array.
    Returns the Annualized Sharpe Ratio (Fitness).
    """
    df = df_data.copy()
    
    # Unpack parameters: [C, L, E, S, P, M]
    center, leverage, exponent, sl_percent, reentry_prox, sma_period = individual
    sma_period = int(sma_period) # Ensure SMA is an integer

    # 1. Calculate SMA & Raw Daily Returns
    df[f'SMA_{sma_period}'] = df['Close'].rolling(window=sma_period).mean()
    df['Daily_Return_Raw'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Look-ahead Prevention & Core Indicators ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA'] = df[f'SMA_{sma_period}'].shift(1)
    df['Proximity_to_SMA'] = np.abs((df['Yesterday_Close'] - df['Yesterday_SMA']) / df['Yesterday_SMA'])
    
    # Drop initial NaNs created by the SMA and shift operations
    df = df.dropna().copy()
    
    if df.empty:
        return -np.inf

    # Initialize Series for returns
    strategy_returns = pd.Series(index=df.index, dtype=float)
    sl_cooldown = False 
    
    # Iterate through the DataFrame for day-by-day logic
    for i in range(len(df)):
        index = df.index[i]
        
        entry_price = df.loc[index, 'Yesterday_Close']
        yesterday_sma = df.loc[index, 'Yesterday_SMA']
        proximity = df.loc[index, 'Proximity_to_SMA']
        
        # --- 1. Determine Base Position and Direction ---
        direction = np.where(entry_price > yesterday_sma, 1, -1)
        distance_d = np.abs((entry_price - yesterday_sma) / yesterday_sma)
        
        # Multiplier (M) calculation
        distance_scaler = 1.0 / center
        scaled_distance = distance_d * distance_scaler
        epsilon = 1e-10 
        
        # Denominator: (1/Scaled_Dist) + Scaled_Dist - 1
        denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
        
        # Multiplier = 1 / (Denominator)^E
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
        
        # Store the daily return
        strategy_returns[index] = daily_return * position_size_base * leverage

    # Final Sharpe Calculation
    return calculate_sharpe_ratio(strategy_returns.dropna())

# --- 4. Genetic Algorithm Core Functions ---

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
    
    # Combine population and fitness, then sort by fitness (descending)
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    
    # Always take the best individual (Elite)
    parents.append(sorted_population[0][0])
    
    # Use Tournament Selection for the rest
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
            
            # Apply a small Gaussian nudge
            nudge = random.gauss(0, (high - low) * 0.1) 
            mutated_gene = gene + nudge
            
            # Enforce bounds
            mutated_gene = max(low, min(high, mutated_gene))
            
            # Enforce data type (round for integers)
            if dtype == 'int':
                mutated_gene = int(round(mutated_gene))
            
            individual[i] = mutated_gene
    return individual

def run_genetic_algorithm(df):
    """Main GA loop with targeted date slicing."""
    print("\n" + "=" * 80)
    print("STARTING GENETIC ALGORITHM OPTIMIZATION (BULL MARKET WINDOW)")
    print(f"Optimization Period: {OPT_START_DATE} to {OPT_END_DATE}")
    print(f"Target Metric: Annualized Sharpe Ratio | Pop Size: {POPULATION_SIZE} | Gens: {GENERATIONS}")
    print("Parameters being optimized: C, L, E, S, P, SMA Period")
    print("=" * 80)
    
    # 1. Slice data to the specific optimization period
    try:
        df_in_sample = df.loc[OPT_START_DATE:OPT_END_DATE].copy() 
    except KeyError:
        print(f"Error: Data for period {OPT_START_DATE} to {OPT_END_DATE} not fully available or indexed incorrectly. Check start date of fetched data.")
        return None
        
    # 2. Initialize population
    population = initialize_population()
    best_sharpe_overall = -np.inf
    best_individual = None
    
    # 3. Main GA Loop
    for generation in range(GENERATIONS):
        # Calculate fitness for the current population
        fitnesses = []
        for individual in population:
            # Pass the sliced data frame to the fitness function
            fitness = run_strategy_for_fitness(df_in_sample, individual)
            fitnesses.append(fitness)
        
        # Find the best individual in this generation
        current_best_index = np.argmax(fitnesses)
        current_best_sharpe = fitnesses[current_best_index]
        
        if current_best_sharpe > best_sharpe_overall:
            best_sharpe_overall = current_best_sharpe
            best_individual = population[current_best_index]
        
        # Selection
        parents = select_parents(population, fitnesses)
        
        # Create next generation via Crossover and Mutation
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

    # 4. Final results display
    print("\n" + "=" * 80)
    print("GENETIC ALGORITHM OPTIMIZATION COMPLETE")
    print(f"Overall Max Annualized Sharpe Ratio (In-Sample: {OPT_START_DATE} to {OPT_END_DATE}): {best_sharpe_overall:.4f}")
    
    if best_individual:
        # Map indices to names for clean output
        results = {}
        for i, (low, high, dtype, name) in enumerate(PARAM_BOUNDS):
            value = best_individual[i]
            if dtype == 'int':
                value = int(value)
            results[name] = value

        print("\nOptimal Parameters Found (Bull Market Focus):")
        for name, value in results.items():
            print(f"  {name}: {value}")
    print("=" * 80)
    
    return best_individual

# --- Main Execution ---

if __name__ == '__main__':
    # 1. Fetch data
    df_data = fetch_klines(SYMBOL, INTERVAL, START_DATE)
    
    if df_data.empty:
        print("Error: Could not retrieve data. Exiting.")
    else:
        # 2. Run Genetic Algorithm
        optimal_params = run_genetic_algorithm(df_data)
        
        if optimal_params is not None:
             # Map indices to names for final printout
             final_results = {}
             for i, (low, high, dtype, name) in enumerate(PARAM_BOUNDS):
                 value = optimal_params[i]
                 if dtype == 'int':
                     value = int(value)
                 final_results[name] = value
                 
             print("\nOptimization successful. Please save the Bull Market parameters above for the final strategy.")
        else:
             print("\nOptimization failed to find a valid optimal parameter set.")
