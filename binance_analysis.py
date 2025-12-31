import pandas as pd
from flask import Flask, render_template_string
import json

app = Flask(__name__)

# Mapping der technischen Kürzel auf lesbare Namen
COLUMN_MAPPING = {
    "C-TIIDXM-0": "Zeitraum",
    "F-UIDXNOM": "Umsatz (Nom.)",
    "F-UIDXREAL": "Umsatz (Real)",
    "F-BESCHIDX": "Beschäftigte",
    "F-UIDXNAB": "Umsatz Nom. (ber.)",
    "F-UIDXNSB": "Umsatz Nom. (sais.)",
    "F-UIDXRAB": "Umsatz Real (ber.)",
    "F-UIDXRSB": "Umsatz Real (sais.)",
    "F-IDXBLG": "Lohnindex",
    "F-IDXGA": "Arbeitsstunden",
    "C-NACEIDX-0": "NACE Branche"
}

STYLE = """
<style>
    body {
        background-color: #c0c0c0;
        background-image: radial-gradient(#888 1px, transparent 1px);
        background-size: 25px 25px;
        color: #000080;
        font-family: "Courier New", Courier, monospace;
        margin: 20px;
    }
    .win-container {
        border: 3px outset #ffffff;
        padding: 15px;
        background-color: #d4d0c8;
        box-shadow: 5px 5px 0px #444;
    }
    .header-bar {
        background: linear-gradient(90deg, #000080, #1084d0);
        color: white;
        padding: 4px 10px;
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
    }
    h1 {
        text-align: center;
        color: #800000;
        font-size: 1.5em;
        text-transform: uppercase;
        margin: 10px 0;
    }
    .chart-box {
        background-color: #000;
        border: 4px inset #888;
        padding: 10px;
        margin: 20px 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        background-color: #ffffff;
        border: 2px inset #ffffff;
    }
    th {
        background-color: #c0c0c0;
        color: #000;
        padding: 5px;
        border: 1px solid #888;
        font-size: 0.8em;
    }
    td {
        border: 1px solid #dfdfdf;
        padding: 4px;
        font-size: 0.8em;
    }
    .blink { animation: blinker 1s linear infinite; color: #00ff00; }
    @keyframes blinker { 50% { opacity: 0; } }
    footer { text-align: center; font-size: 0.7em; margin-top: 20px; }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>TERMINAL - HANDELSDATEN ÖSTERREICH</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {{ style|safe }}
</head>
<body>
    <div class="win-container">
        <div class="header-bar">
            <span>DATA_REPORTER.EXE</span>
            <span>[X]</span>
        </div>
        
        <h1>Marktanalyse: Österreichischer Handel (Basis 2021)</h1>
        
        <div style="background:#000; color:#0f0; padding:10px; font-size:0.9em; margin-bottom:15px;">
            > SYSTEM STATUS: <span class="blink">SCANNING LIQUIDITY...</span><br>
            > SOURCE: Statistik Austria Open Data Portal<br>
            > RECORDS: {{ row_count }} geladen.
        </div>

        <div class="chart-box">
            <canvas id="mainChart" style="max-height: 300px;"></canvas>
        </div>

        <div style="overflow-x: auto;">
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
        </div>

        <footer>
            PROZESSIERT MIT PYTHON 3.x / FLASK 2.0<br>
            VERBINDUNG ZU STATISTIK.AT HERGESTELLT...
        </footer>
    </div>

    <script>
        const ctx = document.getElementById('mainChart').getContext('2d');
        const chartData = {{ chart_json|safe }};
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: 'Umsatzindex Nominell',
                    data: chartData.turnover,
                    borderColor: '#00ff00',
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    borderWidth: 2,
                    pointRadius: 1,
                    tension: 0.1
                }, {
                    label: 'Beschäftigtenindex',
                    data: chartData.employment,
                    borderColor: '#ff00ff',
                    borderWidth: 2,
                    pointRadius: 1,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: '#333' }, ticks: { color: '#0f0' } },
                    x: { grid: { color: '#333' }, ticks: { color: '#0f0' } }
                },
                plugins: {
                    legend: { labels: { color: '#0f0', font: { family: 'Courier New' } } }
                }
            }
        });
    </script>
</body>
</html>
"""

def load_data():
    try:
        # Lade CSV (erkennt automatisch Komma oder Semikolon)
        try:
            df = pd.read_csv('data.csv', sep=';')
        except:
            df = pd.read_csv('data.csv')
            
        # Spalten umbenennen basierend auf dem Mapping
        df = df.rename(columns=COLUMN_MAPPING)
        
        # Daten für die Chart vorbereiten (letzte 24 Monate für bessere Sichtbarkeit)
        chart_subset = df.head(24) if len(df) > 24 else df
        # Falls Kommas statt Punkte in Zahlen sind:
        for col in ["Umsatz (Nom.)", "Beschäftigte"]:
            if col in chart_subset.columns:
                chart_subset[col] = pd.to_numeric(chart_subset[col].astype(str).str.replace(',', '.'), errors='coerce')

        chart_json = {
            "labels": chart_subset["Zeitraum"].tolist()[::-1],
            "turnover": chart_subset["Umsatz (Nom.)"].tolist()[::-1],
            "employment": chart_subset["Beschäftigte"].tolist()[::-1]
        }

        # Tabellendaten (Top 50 für Web-Übersicht)
        sample = df.head(50)
        return sample.columns.tolist(), sample.values.tolist(), len(df), chart_json
    except Exception as e:
        return ["Fehler"], [[str(e)]], 0, {"labels":[], "turnover":[], "employment":[]}

@app.route('/')
def index():
    cols, rows, count, chart_json = load_data()
    return render_template_string(
        HTML_TEMPLATE, 
        style=STYLE, 
        columns=cols, 
        rows=rows, 
        row_count=count,
        chart_json=json.dumps(chart_json)
    )

if __name__ == '__main__':
    print("Initialisiere Forschungs-Terminal...")
    app.run(host='0.0.0.0', port=8080)