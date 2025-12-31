import pandas as pd
from flask import Flask, render_template_string
import json
import io

app = Flask(__name__)

# CSS für das "Matrix/Militär-Terminal" der 90er
STYLE = """
<style>
    body { 
        background-color: #000b00; 
        color: #00ff41; 
        font-family: 'Courier New', Courier, monospace; 
        margin: 0; 
        overflow-x: hidden;
    }
    .terminal {
        padding: 30px;
        max-width: 900px;
        margin: auto;
        position: relative;
    }
    .scanline {
        width: 100%; height: 3px; background: rgba(0, 255, 65, 0.1);
        position: fixed; top: 0; left: 0; pointer-events: none;
        animation: scan 8s linear infinite;
    }
    @keyframes scan { from { top: 0; } to { top: 100%; } }
    h1 { 
        border: 2px solid #00ff41; 
        padding: 10px; 
        text-align: center; 
        text-shadow: 0 0 5px #00ff41;
        letter-spacing: 5px;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 20px;
    }
    .box {
        border: 1px solid #00ff41;
        padding: 15px;
        background: rgba(0, 255, 65, 0.05);
    }
    .chart-container {
        border: 1px solid #00ff41;
        margin: 20px 0;
        background: #000;
        padding: 10px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        font-size: 0.85em;
    }
    th { background: #00441b; border: 1px solid #00ff41; padding: 10px; }
    td { border: 1px solid #00ff41; padding: 8px; text-align: center; }
    .liquidity-high { color: #fff; text-shadow: 0 0 10px #00ff41; font-weight: bold; }
    .blink { animation: blinker 1s steps(2, start) infinite; }
    @keyframes blinker { to { visibility: hidden; } }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LIQUIDITY TERMINAL V.4791</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {{ style|safe }}
</head>
<body>
    <div class="scanline"></div>
    <div class="terminal">
        <p style="font-size: 0.7em;">LOGGED IN AS: RESEARCH_UNITS_AUT | ACCESS_LEVEL: RESTRICTED</p>
        <h1>E-COMMERCE LIQUIDITY MONITOR</h1>
        
        <div class="stats-grid">
            <div class="box">
                <p>> SEKTOR: <span class="blink">G4791</span></p>
                <p>> DESC: Versand- & Internet-Einzelhandel</p>
                <p>> REGION: Österreich</p>
            </div>
            <div class="box">
                <p>> BASIS_YR: 2021 (Index=100)</p>
                <p>> STATUS: Datenstrom synchronisiert</p>
                <p>> ANALYSE: {{ row_count }} Zeitpunkte gefunden</p>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="liquidChart" style="height: 300px;"></canvas>
        </div>

        <table>
            <thead>
                <tr>
                    <th>ZEITRAUM</th>
                    <th>LIQUIDITÄT (NOM)</th>
                    <th>REAL-WERT</th>
                    <th>PERSONAL-IDX</th>
                </tr>
            </thead>
            <tbody>
                {% for row in rows %}
                <tr>
                    <td>{{ row['Zeitraum'] }}</td>
                    <td class="liquidity-high">{{ row['Umsatz_Nom'] }}</td>
                    <td>{{ row['Umsatz_Real'] }}</td>
                    <td>{{ row['Beschäftigte'] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <p style="margin-top: 20px; font-size: 0.8em; text-align: center;">
            *** INTERPRETATION: Werte > 100 zeigen erhöhte Marktaktivität gegenüber dem Basisjahr 2021 ***
        </p>
    </div>

    <script>
        const ctx = document.getElementById('liquidChart').getContext('2d');
        const d = {{ chart_json|safe }};
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: d.labels,
                datasets: [{
                    label: 'Online-Liquidity Index',
                    data: d.values,
                    borderColor: '#00ff41',
                    backgroundColor: 'rgba(0, 255, 65, 0.2)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 3,
                    pointBackgroundColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: '#00441b' }, ticks: { color: '#00ff41' } },
                    x: { grid: { color: '#00441b' }, ticks: { color: '#00ff41' } }
                },
                plugins: {
                    legend: { labels: { color: '#00ff41', font: { family: 'Courier New' } } }
                }
            }
        });
    </script>
</body>
</html>
"""

def clean_val(val):
    """Bereinigt Präfixe und europäische Zahlenformate."""
    if pd.isna(val): return 0.0
    s = str(val).replace('TIIDX-', '').replace('NACEIDX-', '').replace(',', '.')
    try:
        return float(s)
    except:
        return s

def process():
    try:
        # Lade die CSV (Semikolon oder Tab getrennt basierend auf deinem Snippet)
        df = pd.read_csv('data.csv', sep=None, engine='python', header=None)
        
        # Wir benennen die Spalten nach deinem Schema
        df.columns = [
            "C-TIIDXM-0", "C-NACEIDX-0", "F-UIDXNOM", "F-UIDXREAL", 
            "F-BESCHIDX", "F-UIDXNAB", "F-UIDXNSB", "F-UIDXRAB", 
            "F-UIDXRSB", "F-IDXBLG", "F-IDXBLGAB", "F-IDXGA", "F-IDXGAAB"
        ]

        # Filter auf E-Commerce
        online_df = df[df['C-NACEIDX-0'].astype(str).str.contains('4791')].copy()

        # Daten bereinigen
        online_df['Zeitraum'] = online_df['C-TIIDXM-0'].apply(clean_val)
        online_df['Umsatz_Nom'] = online_df['F-UIDXNOM'].apply(clean_val)
        online_df['Umsatz_Real'] = online_df['F-UIDXREAL'].apply(clean_val)
        online_df['Beschäftigte'] = online_df['F-BESCHIDX'].apply(clean_val)

        # Sortieren für Chart (Zeitstrahl)
        online_df = online_df.sort_values(by='Zeitraum')

        chart_data = {
            "labels": online_df['Zeitraum'].tolist(),
            "values": online_df['Umsatz_Nom'].tolist()
        }

        return online_df.to_dict('records'), len(online_df), chart_data
    except Exception as e:
        print(f"ERROR: {e}")
        return [], 0, {"labels": [], "values": []}

@app.route('/')
def index():
    rows, count, chart_json = process()
    return render_template_string(
        HTML_TEMPLATE, 
        style=STYLE, 
        rows=rows, 
        row_count=count,
        chart_json=json.dumps(chart_json)
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)