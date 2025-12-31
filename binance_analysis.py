import pandas as pd
from flask import Flask, render_template_string
import os

app = Flask(__name__)

# CSS for that authentic 90s scientific aesthetic
STYLE = """
<style>
    body {
        background-color: #c0c0c0; /* Classic Windows Gray */
        background-image: radial-gradient(#d1d1d1 1px, transparent 1px);
        background-size: 20px 20px;
        color: #000080; /* Navy Blue Text */
        font-family: "Courier New", Courier, monospace;
        margin: 40px;
    }
    .container {
        border: 3px outset #ffffff;
        padding: 20px;
        background-color: #d4d0c8;
    }
    h1 {
        text-align: center;
        text-decoration: underline;
        color: #800000; /* Maroon */
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        background-color: #ffffff;
        border: 2px inset #ffffff;
        margin-top: 20px;
    }
    th {
        background-color: #000080;
        color: #ffffff;
        padding: 8px;
        border: 1px solid #c0c0c0;
        font-size: 0.9em;
    }
    td {
        border: 1px solid #c0c0c0;
        padding: 6px;
        font-size: 0.85em;
    }
    tr:nth-child(even) {
        background-color: #f0f0f0;
    }
    .header-info {
        border: 1px dashed #000000;
        padding: 10px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .blink {
        animation: blinker 1s linear infinite;
        color: red;
        font-weight: bold;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    .marquee-container {
        background: #000;
        color: #0f0;
        padding: 5px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    footer {
        margin-top: 30px;
        font-size: 0.7em;
        text-align: center;
        color: #666;
    }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AUSTRIA DATA PORTAL - ECONOMIC INDICES 2021</title>
    {{ style|safe }}
</head>
<body>
    <div class="container">
        <div class="marquee-container">
            <marquee scrollamount="5">*** SYSTEM READY ... LOADING SECTORAL LIQUIDITY DATA FOR AUSTRIA ... SOURCE: STATISTIK AUSTRIA ***</marquee>
        </div>
        
        <h1>Data Overview: Konjunkturindizes Handel</h1>
        
        <div class="header-info">
            REPORT ID: AUT-ECO-2021-X<br>
            STATUS: <span class="blink">LIVE DATA STREAM</span><br>
            RECORDS FOUND: {{ row_count }}
        </div>

        <p>This terminal displays the current trade indices (Basis 2021). High values in turnover indices generally correlate with high product liquidity within that specific NACE sector.</p>

        <table>
            <thead>
                <tr>
                    {% for col in columns %}
                    <th>{{ col }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in rows %}
                <tr>
                    {% for cell in row %}
                    <td>{{ cell }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <footer>
            OPTIMIZED FOR NETSCAPE NAVIGATOR 4.0<br>
            &copy; 1998 SCIENTIFIC DATA SERVICES - AUSTRIA DIVISION
        </footer>
    </div>
</body>
</html>
"""

def load_data():
    try:
        # Load the CSV. We use ';' as it is common for Austrian/European CSVs.
        # If it fails, we try a standard comma.
        try:
            df = pd.read_csv('data.csv', sep=';')
        except:
            df = pd.read_csv('data.csv')
            
        # Limiting to top 100 for readability in the overview
        data_sample = df.head(100)
        return data_sample.columns.tolist(), data_sample.values.tolist(), len(df)
    except Exception as e:
        return ["Error"], [[str(e)]], 0

@app.route('/')
def index():
    cols, rows, count = load_data()
    return render_template_string(
        HTML_TEMPLATE, 
        style=STYLE, 
        columns=cols, 
        rows=rows, 
        row_count=count
    )

if __name__ == '__main__':
    # Running on 0.0.0.0 to be publicly accessible on port 8080
    print("Initializing Scientific Data Server...")
    app.run(host='0.0.0.0', port=8080)