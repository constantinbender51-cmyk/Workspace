import http.server
import socketserver
import urllib.request
import json
import datetime
import time

# Configuration
PORT = 8080
PAIR = "XBTUSD"  # Bitcoin/USD
# Kraken API Interval in minutes: 10080 = 1 week
INTERVAL = 10080 

def fetch_and_resample_data():
    """
    Fetches Weekly OHLC data from Kraken and resamples it to Monthly Close.
    """
    url = f"https://api.kraken.com/0/public/OHLC?pair={PAIR}&interval={INTERVAL}"
    
    try:
        print(f"Fetching weekly data from {url}...")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
        if data.get('error'):
            print(f"API Error: {data['error']}")
            return [], []

        # Get the result list (key is dynamic, e.g., XXBTZUSD)
        result_key = list(data['result'].keys())[0]
        ohlc_data = data['result'][result_key]

        # Resampling Logic: Weekly -> Monthly
        # Dictionary to store { 'YYYY-MM': close_price }
        monthly_map = {}

        for entry in ohlc_data:
            # Entry: [time, open, high, low, close, vwap, volume, count]
            timestamp = entry[0]
            close_price = float(entry[4])
            
            # Convert timestamp to date object
            dt = datetime.datetime.fromtimestamp(timestamp)
            
            # Create a key for the month (e.g., "2023-10")
            month_key = dt.strftime('%Y-%m')
            
            # Since data is chronological, writing to the map repeatedly 
            # ensures the *last* weekly entry for a month becomes the representative close.
            monthly_map[month_key] = close_price

        # Extract sorted lists for the chart
        labels = list(monthly_map.keys())
        prices = list(monthly_map.values())

        return labels, prices

    except Exception as e:
        print(f"Failed to process data: {e}")
        return [], []

class ChartHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            labels, data = fetch_and_resample_data()

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Kraken {PAIR} Monthly Resample</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
                    .container {{ max-width: 1000px; margin: 0 auto; }}
                    h1 {{ text-align: center; color: #fff; font-weight: 300; letter-spacing: 1px; }}
                    .subtitle {{ text-align: center; color: #888; font-size: 0.9em; margin-bottom: 20px; }}
                    .chart-wrapper {{ background: #1e1e1e; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
                    .chart-container {{ position: relative; height: 60vh; width: 100%; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{PAIR} Monthly Close</h1>
                    <div class="subtitle">Source: Kraken API (Weekly Data Resampled)</div>
                    <div class="chart-wrapper">
                        <div class="chart-container">
                            <canvas id="myChart"></canvas>
                        </div>
                    </div>
                </div>

                <script>
                    const ctx = document.getElementById('myChart').getContext('2d');
                    
                    // Create gradient
                    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
                    gradient.addColorStop(0, 'rgba(54, 162, 235, 0.5)');
                    gradient.addColorStop(1, 'rgba(54, 162, 235, 0.0)');

                    const myChart = new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: {json.dumps(labels)},
                            datasets: [{{
                                label: 'Monthly Close (USD)',
                                data: {json.dumps(data)},
                                borderColor: '#36a2eb',
                                backgroundColor: gradient,
                                borderWidth: 2,
                                pointRadius: 3,
                                pointBackgroundColor: '#36a2eb',
                                fill: true,
                                tension: 0.3
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ labels: {{ color: '#ccc' }} }},
                                tooltip: {{
                                    mode: 'index',
                                    intersect: false,
                                    backgroundColor: 'rgba(0,0,0,0.8)',
                                    titleColor: '#fff',
                                    bodyColor: '#fff'
                                }}
                            }},
                            scales: {{
                                x: {{
                                    grid: {{ color: '#333' }},
                                    ticks: {{ color: '#888' }}
                                }},
                                y: {{
                                    grid: {{ color: '#333' }},
                                    ticks: {{ color: '#888' }}
                                }}
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ChartHandler) as httpd:
        print(f"Serving Resampled Kraken Chart at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()

if __name__ == "__main__":
    run_server()