import pandas as pd
from flask import Flask, render_template_string
import json

app = Flask(__name__)

# Extended Mapping for KJPC2 (ÖCPA 2-Steller) Industry Divisions
INDUSTRY_MAP = {
    "01": "Landwirtschaft & Jagd",
    "02": "Forstwirtschaft",
    "03": "Fischerei & Aquakultur",
    "05": "Kohlenbergbau",
    "08": "Steine & Erden (Bergbau)",
    "10": "Nahrungsmittel (Lebensmittelproduktion)",
    "11": "Getränke (Spirituosen, Wein, Bier, Softdrinks)",
    "12": "Tabakwaren",
    "13": "Textilien (Stoffe, Garne)",
    "14": "Bekleidung (Kleidung, Kürschnerei)",
    "15": "Lederwaren & Schuhe",
    "16": "Holz-, Flecht- & Korbwaren (ohne Möbel)",
    "17": "Papier, Pappe & Waren daraus",
    "18": "Druckerzeugnisse & Vervielfältigung",
    "19": "Kokerei & Mineralölverarbeitung",
    "20": "Chemische Erzeugnisse (Kosmetik, Reinigungsmittel)",
    "21": "Pharmazeutische Erzeugnisse (Medizin)",
    "22": "Gummi- & Kunststoffwaren",
    "23": "Glas, Keramik, Steine & Erden (Verarbeitung)",
    "24": "Metallerzeugung & -bearbeitung",
    "25": "Metallerzeugnisse (Werkzeuge, Waffen, Behälter)",
    "26": "Datenverarbeitungsgeräte, elektronische & optische Erzeugnisse",
    "27": "Elektrische Ausrüstungen (Batterien, Kabel, Leuchten)",
    "28": "Maschinenbau (Industriemaschinen)",
    "29": "Kraftwagen & Kraftwagenteile",
    "30": "Sonstiger Fahrzeugbau (Schiffe, Flugzeuge, Bahn)",
    "31": "Möbel (Wohn-, Büro- & Küchenmöbel)",
    "32": "Schmuck, Musikinstrumente, Sportgeräte, Spielwaren",
    "33": "Reparatur & Installation von Maschinen",
    "35": "Energieversorgung (Strom, Gas, Fernwärme)",
    "38": "Abfallentsorgung & Recycling",
    "41": "Hochbau (Gebäudeerrichtung)",
    "42": "Tiefbau (Straßen, Schienen, Leitungen)",
    "43": "Vorbereitende Baustellenarbeiten & Ausbau",
    "44": "Spezialisierte Bautätigkeiten",
    "46": "Großhandel (Handelsvermittlung)"
}

STYLE = """
<style>
    body { background-color: #000; color: #00ff00; font-family: 'Courier New', Courier, monospace; margin: 0; padding: 20px; }
    .terminal { border: 2px solid #00ff00; background: #050505; padding: 20px; box-shadow: 0 0 30px #003300; }
    .header { border-bottom: 2px double #00ff00; margin-bottom: 20px; padding-bottom: 10px; }
    .row-item { display: grid; grid-template-columns: 100px 1fr 180px 120px; gap: 10px; padding: 8px; border-bottom: 1px solid #003300; }
    .row-item:hover { background: #002200; }
    .label-header { font-weight: bold; color: #fff; background: #003300; padding: 5px; }
    .code { color: #888; }
    .industry { font-weight: bold; color: #00ff00; }
    .value { text-align: right; color: #ffff00; font-weight: bold; }
    .status { text-align: center; font-size: 0.8em; }
    .blink { animation: blinker 1s steps(2, start) infinite; color: red; }
    @keyframes blinker { to { visibility: hidden; } }
    .high-liquidity { background-color: rgba(255, 255, 0, 0.1); }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AUSTRIA_INDUSTRY_DECODER_V2</title>
    {{ style|safe }}
</head>
<body>
    <div class="terminal">
        <div class="header">
            <h2 style="margin:0;">[SYSTEM_DECODE] INDUSTRY & PRODUCT ANALYSIS</h2>
            <div style="font-size: 0.8em; margin-top:5px;">DATA SOURCE: STATISTIK AUSTRIA | PARAMETER: F-PROD4 (ABSATZWERT)</div>
        </div>

        <div class="row-item label-header">
            <div>CODE</div>
            <div>INDUSTRIE / PRODUKTGRUPPE</div>
            <div style="text-align:right;">ABSATZWERT (EUR)</div>
            <div style="text-align:center;">STATUS</div>
        </div>

        {% for row in data %}
        <div class="row-item {{ 'high-liquidity' if row.val > 10000000 else '' }}">
            <div class="code">{{ row.code }}</div>
            <div class="industry">{{ row.name }}</div>
            <div class="value">{{ "{:,.0f}".format(row.val).replace(',', '.') }}</div>
            <div class="status">
                {% if row.val > 10000000 %}
                <span class="blink">TOP-SELLER</span>
                {% else %}
                STABIL
                {% endif %}
            </div>
        </div>
        {% endfor %}

        <div style="margin-top: 20px; border-top: 1px solid #00ff00; padding-top: 10px; font-size: 0.8em;">
            > ANALYSE AKTIV: Suche nach Nischen in Sektoren 10, 20 und 26...<br>
            > HINWEIS: Hohe Millionenwerte in Sektoren 35/43 deuten auf Infrastruktur hin. Konsumgüter finden sich bevorzugt in Sektoren 10-32.
        </div>
    </div>
</body>
</html>
"""

def clean_code(raw):
    """Extrahiert die reine Zahl aus dem Code KJPC2-XX oder NACEIDX-XX."""
    s = str(raw).upper()
    s = s.replace('KJPC2-', '').replace('NACEIDX-', '').replace('PCM2-', '')
    return s.strip()

def clean_val(raw):
    if pd.isna(raw): return 0.0
    s = str(raw).replace(' EUR', '').replace('.', '').replace(',', '.')
    try: return float(s)
    except: return 0.0

def process():
    try:
        # Lade die CSV (Automatische Erkennung von Trennern)
        df = pd.read_csv('data.csv', sep=None, engine='python', header=None)
        
        # Annahme: Spalte 0 ist der Code, Spalte 1 ist der Wert (basierend auf deinem Snippet)
        # Wir passen dies dynamisch an, falls Spaltennamen vorhanden sind
        results = []
        for _, row in df.iterrows():
            raw_code = clean_code(row[0])
            raw_val = clean_val(row[1])
            
            if raw_code.isdigit() or raw_code in INDUSTRY_MAP:
                results.append({
                    "code": f"KJPC2-{raw_code}",
                    "name": INDUSTRY_MAP.get(raw_code, "Unbekannter Sektor"),
                    "val": raw_val
                })
        
        # Sortieren nach Absatzwert
        results = sorted(results, key=lambda x: x['val'], reverse=True)
        return results
    except Exception as e:
        # Fallback mit deinen geposteten Daten für die Demo
        demo_data = [
            ("35", 43983933), ("43", 16737792), ("28", 16482587), ("24", 15855018),
            ("10", 15402229), ("25", 11759429), ("46", 11147452), ("29", 10997720),
            ("41", 10122694), ("27", 8575699), ("11", 8076775), ("26", 7487654),
            ("42", 7334836), ("16", 7014997), ("20", 6656164), ("33", 6154166),
            ("22", 5358848), ("17", 4947561), ("44", 4828166), ("21", 4569305),
            ("23", 4319086), ("19", 4081418), ("38", 3981756)
        ]
        results = []
        for c, v in demo_data:
            results.append({"code": f"KJPC2-{c}", "name": INDUSTRY_MAP.get(c, "Diverses"), "val": v})
        return results

@app.route('/')
def index():
    data = process()
    return render_template_string(HTML_TEMPLATE, style=STYLE, data=data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)