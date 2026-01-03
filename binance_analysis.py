import http.server
import socketserver
import urllib.request
import json
import datetime
import time

# Configuration
PORT = 8080
PAIR = "XBTUSD"  # Bitcoin/USD
INTERVAL = 1440  # 1440 minutes = 1 day (Kraken's API is reliable with daily; we can visualize monthly trends from this)

def fetch_kraken_data():
    """
    Fetches OHLC data from Kraken public API using standard urllib.
    Returns lists of labels (dates) and data (close prices).
    """
    url = f"https://api.kraken.com/0/public/OHLC?pair={PAIR}&interval={INTERVAL}"
    
    try:
        print(f"Fetching data from {url}...")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
        if data.get('error'):
            print(f"API Error: {data['error']}")
            return [], []

        # Kraken returns a dictionary where keys are pair names. 
        # We find the result key that matches our pair (e.g., XXBTZUSD)
        result_key = list(data['result'].keys())[0]
        ohlc_data = data['result'][result_key]

        # OHLC Structure: [time, open, high, low, close, vwap, volume, count]
        # We need Time (0) and Close (4)
        labels = []
        prices = []

        for entry in ohlc_data:
            # Convert timestamp to readable date
            dt = datetime.datetime.fromtimestamp(entry[0])
            # Format as YYYY-MM-DD
            date_str = dt.strftime('%Y-%m-%d')
            
            labels.append(date_str)
            prices.append(float(entry[4]))

        return labels, prices

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return [], []

class ChartHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # We only handle the root path
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            # Fetch fresh data on every request
            labels, data = fetch_kraken_data()

            # HTML Template with Chart.js
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Kraken {PAIR} OHLC</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{ font-family: sans-serif; background: #1e1e1e; color: #ddd; margin: 0; padding: 20px; }}
                    .container {{ max-width: 1000px; margin: 0 auto; }}
                    h1 {{ text-align: center; color: #fff; }}
                    .chart-container {{ position: relative; height: 60vh; width: 100%; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{PAIR} Close Prices (Daily/Monthly Trend)</h1>
                    <div class="chart-container">
                        <canvas id="myChart"></canvas>
                    </div>
                </div>

                <script>
                    const ctx = document.getElementById('myChart').getContext('2d');
                    const myChart = new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: {json.dumps(labels)},
                            datasets: [{{
                                label: 'Close Price (USD)',
                                data: {json.dumps(data)},
                                borderColor: 'rgba(75, 192, 192, 1)',
                                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                borderWidth: 2,
                                pointRadius: 0, // Hide points for cleaner look on large datasets
                                pointHoverRadius: 5,
                                fill: true,
                                tension: 0.1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            interaction: {{
                                mode: 'index',
                                intersect: false,
                            }},
                            scales: {{
                                x: {{
                                    grid: {{ color: '#333' }},
                                    ticks: {{ color: '#aaa' }}
                                }},
                                y: {{
                                    grid: {{ color: '#333' }},
                                    ticks: {{ color: '#aaa' }}
                                }}
                            }},
                            plugins: {{
                                legend: {{ labels: {{ color: '#fff' }} }}
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            # 404 for anything else
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

def run_server():
    # Allow reuse of address to prevent "Address already in use" errors on restart
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), ChartHandler) as httpd:
        print(f"Serving Kraken OHLC chart at http://localhost:{PORT}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()