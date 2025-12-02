import pandas as pd
import gdown
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, render_template_string

app = Flask(__name__)

# HTML template for the webpage
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Exoplanet Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .plot-container {
            text-align: center;
            margin: 30px 0;
        }
        .plot-img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .stats {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin-top: 30px;
        }
        .stats h2 {
            color: #333;
            margin-top: 0;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .data-table th, .data-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .data-table th {
            background-color: #f2f2f2;
        }
        .data-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .note {
            font-style: italic;
            color: #666;
            margin-top: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Exoplanet Visualization</h1>
        <div class="subtitle">Distance (sy_dist) vs Equilibrium Temperature (pl_eqt)</div>
        
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plot_data }}" alt="Exoplanet Distance vs Temperature Plot" class="plot-img">
        </div>
        
        <div class="stats">
            <h2>Dataset Information</h2>
            <p><strong>Total planets plotted:</strong> {{ total_planets }}</p>
            <p><strong>Data source:</strong> NASA Exoplanet Archive</p>
            <p><strong>X-axis:</strong> Distance from Earth (parsecs) - sy_dist</p>
            <p><strong>Y-axis:</strong> Equilibrium Temperature (Kelvin) - pl_eqt</p>
            
            {% if sample_data is defined and sample_data|length > 0 %}
            <h3>Sample Data (First 10 planets)</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Planet Name</th>
                        <th>Distance (parsecs)</th>
                        <th>Temperature (K)</th>
                        <th>Host Star</th>
                    </tr>
                </thead>
                <tbody>
                    {% for planet in sample_data %}
                    <tr>
                        <td>{{ planet.pl_name if planet.pl_name else 'N/A' }}</td>
                        <td>{{ "%.2f"|format(planet.sy_dist) if planet.sy_dist == planet.sy_dist else 'N/A' }}</td>
                        <td>{{ "%.0f"|format(planet.pl_eqt) if planet.pl_eqt == planet.pl_eqt else 'N/A' }}</td>
                        <td>{{ planet.hostname if planet.hostname else 'N/A' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
        </div>
        
        <div class="note">
            Note: Only planets with both distance and temperature data are shown in the plot.
            Missing values are excluded from visualization.
        </div>
    </div>
</body>
</html>
'''

def load_exoplanet_data():
    """Load the exoplanet data from CSV file"""
    output_file = 'exoplanets.csv'
    
    # If file doesn't exist, download it
    if not os.path.exists(output_file):
        print(f'Downloading {output_file} from Google Drive...')
        file_id = '1pBl_UvRgrCuk6cV06y7-R6gRNB-mr0in'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_file, quiet=False)
    
    if os.path.exists(output_file):
        print(f'Loading {output_file}...')
        
        # Try different methods to load the CSV with comments
        df = None
        
        # First try with comment character
        for comment_char in ['#', ';', '!', '//', '%', '*']:
            try:
                df = pd.read_csv(output_file, comment=comment_char)
                if len(df) > 0 and len(df.columns) > 0:
                    print(f'Successfully loaded with comment={comment_char}: {len(df)} rows')
                    break
            except Exception as e:
                continue
        
        # If that didn't work, try skipping rows
        if df is None or len(df) == 0:
            try:
                df = pd.read_csv(output_file, skiprows=96)
                print(f'Loaded by skipping 96 rows: {len(df)} rows')
            except Exception as e:
                print(f'Error loading with skiprows: {e}')
        
        # Final fallback
        if df is None or len(df) == 0:
            try:
                df = pd.read_csv(output_file, engine='python', on_bad_lines='skip')
                print(f'Loaded with flexible parsing: {len(df)} rows')
            except Exception as e:
                print(f'Error loading with flexible parsing: {e}')
                return None
        
        return df
    else:
        print(f'Error: File {output_file} not found')
        return None


def create_plot(df):
    """Create a scatter plot of distance vs temperature"""
    # Filter out rows with missing values for the columns we need
    plot_df = df.dropna(subset=['sy_dist', 'pl_eqt']).copy()
    
    if len(plot_df) == 0:
        print("No data available for plotting (missing sy_dist or pl_eqt values)")
        return None
    
    print(f"Plotting {len(plot_df)} planets with both distance and temperature data")
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(
        plot_df['sy_dist'], 
        plot_df['pl_eqt'], 
        alpha=0.6, 
        c=plot_df['pl_eqt'],  # Color by temperature
        cmap='viridis', 
        s=50  # Marker size
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Equilibrium Temperature (K)', fontsize=12)
    
    # Set labels and title
    plt.xlabel('Distance from Earth (parsecs)', fontsize=14)
    plt.ylabel('Equilibrium Temperature (K)', fontsize=14)
    plt.title('Exoplanets: Distance vs Equilibrium Temperature', fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Use log scale for distance if data spans large range
    if plot_df['sy_dist'].max() / plot_df['sy_dist'].min() > 100:
        plt.xscale('log')
        plt.xlabel('Distance from Earth (parsecs, log scale)', fontsize=14)
    
    # Use log scale for temperature if data spans large range
    if plot_df['pl_eqt'].max() / plot_df['pl_eqt'].min() > 100:
        plt.yscale('log')
        plt.ylabel('Equilibrium Temperature (K, log scale)', fontsize=14)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Encode the plot as base64 for HTML embedding
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    
    return plot_data, plot_df


@app.route('/')
def index():
    """Main route that displays the plot"""
    # Load the data
    df = load_exoplanet_data()
    
    if df is None:
        return "Error: Could not load exoplanet data"
    
    # Check if required columns exist
    required_cols = ['sy_dist', 'pl_eqt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return f"Error: Missing required columns: {missing_cols}. Available columns: {list(df.columns)}"
    
    # Create the plot
    plot_result = create_plot(df)
    
    if plot_result is None:
        return "Error: Could not create plot (no valid data available)"
    
    plot_data, plot_df = plot_result
    
    # Prepare sample data for display
    sample_data = []
    if 'pl_name' in df.columns and 'hostname' in df.columns:
        # Get first 10 planets with both distance and temperature
        sample_df = plot_df.head(10)
        sample_data = sample_df[['pl_name', 'sy_dist', 'pl_eqt', 'hostname']].to_dict('records')
    
    # Render the HTML template with the plot and data
    return render_template_string(
        HTML_TEMPLATE,
        plot_data=plot_data,
        total_planets=len(plot_df),
        sample_data=sample_data
    )


if __name__ == '__main__':
    print("Starting Exoplanet Visualization Server on port 8080...")
    print("Open your browser and navigate to: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
