import pandas as pd
from flask import Flask, render_template_string
import json

app = Flask(__name__)

# Mapping für die wichtigsten NACE-Handelsklassen in Österreich
NACE_MAP = {
    "4711": "Supermärkte / Lebensmittel",
    "4719": "Warenhäuser (Non-Food)",
    "4741": "Computer / Software",
    "4751": "Textilien",
    "4764": "Sportartikel",
    "4771": "Bekleidung",
    "4791": "Online-Handel / Versand",
    "4511": "Auto-Handel"
}

STYLE = """
<style>
    body { background-color: #f0f0f0; color: #333; font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; }
    .header { background: #003366; color: white; padding: 20px; text-align: center; border-bottom: 5px solid #ffcc00; }
    .container { max-width: 1000px; margin: 20px auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .ranking-card { border-left: 5px solid #003366; background: #f9f9f9; padding: 15px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
    .index-val { font-size: 1.5em; font-weight: bold; color: #003366; }
    .label { font-weight: bold; color: #666; }
    .recommendation { background: #e7f3fe; border: 1px solid #b6d4fe; padding: 15px; border-radius: 5px; margin-top: 20px; }
    .tag { background: #ffcc00; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Markt-Liquidität Österreich</title>
    {{ style|safe }}
</head>
<body>
    <div class="header">
        <h1>Produkt-Kategorien Ranking</h1>
        <p>Welche Branche bewegt in Österreich das meiste Kapital?</p>
    </div>
    
    <div class="container">
        <h2>Aktuelle Liquiditäts-Rangliste (Umsatzindex)</h2>
        <p>Ein Index über 100 bedeutet: Die Branche ist aktiver als im Durchschnitt von 2021.</p>
        
        {% for item in ranking %}
        <div class="ranking-card">
            <div>
                <span class="tag">NACE {{ item.code }}</span>
                <div style="font-size: 1.1em; margin-top: 5px;">{{ item.name }}</div>
            </div>
            <div class="index-val">{{ item.value }}</div>
        </div>
        {% endfor %}

        <div class="recommendation">
            <strong>🚀 Strategische Empfehlung:</strong><br>
            Konzentriere dich auf Sektoren, die einen <strong>Umsatzindex (Nominal) > 110</strong> haben. 
            Diese Branchen sind derzeit "liquide" – das Geld der Konsumenten fließt dort aktiv. 
            Wenn der Online-Handel (4791) führt, ist ein reines E-Commerce-Modell am sinnvollsten.
        </div>

        <div style="margin-top:30px; font-size: 0.9em; color: #888;">
            * Quelle: Statistik Austria | Basierend auf deiner data.csv
        </div>
    </div>
</body>
</html>
"""

def get_ranking():
    try:
        df = pd.read_csv('data.csv', sep=None, engine='python', header=None)
        # Spalten-Indizes basierend auf deinem Snippet
        # 0: Zeitraum, 1: NACE, 2: Umsatz_Nom
        
        # Nur den aktuellsten Monat nehmen
        latest_month = df[0].iloc[0]
        current_data = df[df[0] == latest_month]
        
        results = []
        for _, row in current_data.iterrows():
            raw_code = str(row[1]).replace('NACEIDX-', '')
            val = str(row[2]).replace(',', '.')
            try:
                val_float = float(val)
                results.append({
                    "code": raw_code,
                    "name": NACE_MAP.get(raw_code, "Sonstiger Handel / Spezialsortiment"),
                    "value": val_float
                })
            except:
                continue
        
        # Sortieren nach höchstem Umsatzindex (Liquidität)
        return sorted(results, key=lambda x: x['value'], reverse=True)
    except Exception as e:
        return [{"code": "ERR", "name": str(e), "value": 0}]

@app.route('/')
def index():
    ranking = get_ranking()
    return render_template_string(HTML_TEMPLATE, style=STYLE, ranking=ranking)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)