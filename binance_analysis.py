import pandas as pd
from flask import Flask, render_template_string
import json

app = Flask(__name__)

# Mapping der ÖCPA 2-Steller Codes auf lesbare Produktgruppen
OCPA_MAP = {
    "10": "Lebensmittel",
    "11": "Getränke",
    "13": "Textilien",
    "14": "Bekleidung",
    "15": "Lederwaren & Schuhe",
    "16": "Holzwaren (ohne Möbel)",
    "20": "Chemische Erzeugnisse (Kosmetik/Reiniger)",
    "21": "Pharmazeutische Erzeugnisse",
    "26": "Elektronik & Optik",
    "27": "Elektrische Ausrüstungen",
    "28": "Maschinenbau",
    "31": "Möbel",
    "32": "Schmuck, Musikinstrumente, Sportgeräte"
}

STYLE = """
<style>
    body { background-color: #000; color: #00ff00; font-family: 'Courier New', monospace; margin: 0; padding: 20px; }
    .window { border: 2px solid #00ff00; background: #050505; padding: 20px; box-shadow: 0 0 20px #004400; }
    .header { border-bottom: 2px solid #00ff00; margin-bottom: 20px; padding-bottom: 10px; color: #fff; }
    .summary-box { border: 1px dashed #00ff00; padding: 15px; margin-bottom: 20px; background: rgba(0,255,0,0.05); }
    .chart-box { background: #111; border: 1px solid #333; margin: 20px 0; padding: 10px; height: 300px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    th { background: #003300; border: 1px solid #00ff00; padding: 8px; text-align: left; }
    td { border: 1px solid #004400; padding: 8px; }
    .highlight { color: #fff; font-weight: bold; }
    .blink { animation: blinker 1s steps(2, start) infinite; }
    @keyframes blinker { to { visibility: hidden; } }
    .hot-indicator { background: #00ff00; color: #000; padding: 2px 5px; font-weight: bold; font-size: 0.7em; margin-left: 10px; }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ÖCPA_LIQUIDITY_ANALYZER</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {{ style|safe }}
</head>
<body>
    <div class="window">
        <div class="header">
            <div style="font-size: 0.7em; float: right;">LEVEL: ANALYST</div>
            <h2>[ÖCPA-RADAR] BESTSELLER-IDENTIFIKATION</h2>
        </div>

        <div class="summary-box">
            <span class="blink">>>></span> ANALYSE-PARAMETER: Spalte 'F-PROD4' (Abgesetzte Produktion)<br>
            <span class="blink">>>></span> BEDEUTUNG: Hoher AP-Wert = Hoher Marktabsatz (Liquidität)<br>
            <span class="blink">>>></span> FILTER: Wertauswahl (Euro-Beträge)
        </div>

        <div class="chart-box">
            <canvas id="apChart"></canvas>
        </div>

        <table>
            <thead>
                <tr>
                    <th>CODE</th>
                    <th>PRODUKTGRUPPE</th>
                    <th>ABSATZWERT (AP)</th>
                    <th>STATUS</th>
                </tr>
            </thead>
            <tbody>
                {% for row in data %}
                <tr>
                    <td>{{ row.code }}</td>
                    <td class="highlight">{{ row.name }}</td>
                    <td>{{ "{:,.0f}".format(row.val).replace(',', '.') }} EUR</td>
                    <td>
                        {% if row.val > avg_val %}
                        <span class="hot-indicator">TOP-SELLER</span>
                        {% else %}
                        <span>STABIL</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <script>
        const ctx = document.getElementById('apChart').getContext('2d');
        const d = {{ chart_json|safe }};
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: d.labels,
                datasets: [{
                    label: 'Abgesetzte Produktion (Absatz in EUR)',
                    data: d.values,
                    backgroundColor: '#00ff00',
                    borderColor: '#fff',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: '#222' }, ticks: { color: '#00ff00' } },
                    y: { grid: { color: '#222' }, ticks: { color: '#00ff00' } }
                },
                plugins: {
                    legend: { labels: { color: '#00ff00', font: { family: 'Courier New' } } }
                }
            }
        });
    </script>
</body>
</html>
"""

def clean_num(val):
    if pd.isna(val): return 0.0
    s = str(val).replace(',', '.')
    try: return float(s)
    except: return 0.0

def process():
    try:
        # Lade CSV
        df = pd.read_csv('data.csv', sep=None, engine='python')
        
        # Spaltennamen-Mapping zur Sicherheit
        # Wir nutzen die vom User genannten Header
        col_ap = "F-PROD4"
        col_code = "C-PCM2-0"
        
        # Bereinigung der Werte
        df[col_ap] = df[col_ap].apply(clean_num)
        df[col_code] = df[col_code].astype(str).str.replace('PCM2-', '')

        # Gruppierung nach Produktgruppe (ÖCPA 2-Steller)
        grouped = df.groupby(col_code)[col_ap].sum().reset_index()
        
        # Mapping der Namen
        grouped['name'] = grouped[col_code].map(lambda x: OCPA_MAP.get(x, f"Sektor {x}"))
        
        # Sortieren nach Absatzwert
        grouped = grouped.sort_values(by=col_ap, ascending=False)
        
        avg = grouped[col_ap].mean()
        
        results = []
        for _, r in grouped.iterrows():
            results.append({"code": r[col_code], "name": r['name'], "val": r[col_ap]})
            
        chart_data = {
            "labels": [r['name'] for r in results[:10]],
            "values": [r['val'] for r in results[:10]]
        }
        
        return results, avg, chart_data
    except Exception as e:
        print(f"Error: {e}")
        return [], 0, {"labels":[], "values":[]}

@app.route('/')
def index():
    data, avg, chart_json = process()
    return render_template_string(HTML_TEMPLATE, style=STYLE, data=data, avg_val=avg, chart_json=json.dumps(chart_json))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)