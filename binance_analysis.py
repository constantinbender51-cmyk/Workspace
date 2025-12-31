import pandas as pd
from flask import Flask, render_template_string
import json
import re

app = Flask(__name__)

# Extended Mapping for KJPC2 / NACE Industry Divisions
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
    "46": "Großhandel (Handelsvermittlung)",
    "47": "Einzelhandel (Gesamtmarkt)",
    "4791": "Versand- & Internet-Einzelhandel"
}

STYLE = """
<style>
    body { background-color: #000; color: #00ff00; font-family: 'Courier New', Courier, monospace; margin: 0; padding: 20px; }
    .terminal { border: 2px solid #00ff00; background: #050505; padding: 20px; box-shadow: 0 0 30px #003300; }
    .header { border-bottom: 2px double #00ff00; margin-bottom: 20px; padding-bottom: 10px; }
    .row-item { display: grid; grid-template-columns: 120px 1fr 180px 120px; gap: 10px; padding: 8px; border-bottom: 1px solid #003300; }
    .row-item:hover { background: #002200; }
    .label-header { font-weight: bold; color: #fff; background: #003300; padding: 5px; }
    .code { color: #888; font-size: 0.9em; }
    .industry { font-weight: bold; color: #00ff00; }
    .value { text-align: right; color: #ffff00; font-weight: bold; }
    .status { text-align: center; font-size: 0.8em; }
    .blink { animation: blinker 1s steps(2, start) infinite; color: #ff0000; font-weight: bold; }
    @keyframes blinker { to { visibility: hidden; } }
    .high-liquidity { background-color: rgba(255, 255, 0, 0.05); border-left: 4px solid #ffff00; }
    .error-msg { color: #ff0000; border: 1px solid #ff0000; padding: 10px; margin-bottom: 20px; }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AUSTRIA_INDUSTRY_DECODER_V3</title>
    {{ style|safe }}
</head>
<body>
    <div class="terminal">
        <div class="header">
            <h2 style="margin:0;">[SYSTEM_DECODE] INDUSTRY & PRODUCT ANALYSIS</h2>
            <div style="font-size: 0.8em; margin-top:5px;">DATA SOURCE: STATISTIK AUSTRIA | PARAMETER: F-PROD4 / UMSATZINDEX</div>
        </div>

        {% if error %}
        <div class="error-msg">SYSTEM ERROR: {{ error }}</div>
        {% endif %}

        <div class="row-item label-header">
            <div>CODE</div>
            <div>INDUSTRIE / PRODUKTGRUPPE</div>
            <div style="text-align:right;">WERT / ABSATZ</div>
            <div style="text-align:center;">STATUS</div>
        </div>

        {% for row in data %}
        <div class="row-item {{ 'high-liquidity' if row.val > 100 or row.val > 5000000 else '' }}">
            <div class="code">{{ row.raw_code }}</div>
            <div class="industry">{{ row.name }}</div>
            <div class="value">{{ "{:,.2f}".format(row.val).replace(',', 'X').replace('.', ',').replace('X', '.') }}</div>
            <div class="status">
                {% if row.val > 100 or row.val > 10000000 %}
                <span class="blink">HOT</span>
                {% else %}
                STABIL
                {% endif %}
            </div>
        </div>
        {% endfor %}

        <div style="margin-top: 20px; border-top: 1px solid #00ff00; padding-top: 10px; font-size: 0.8em;">
            > ANALYSE AKTIV: Erkenne Codes (NACE/KJPC/ÖCPA)...<br>
            > HINWEIS: Werte > 100 bei Indizes oder hohe Millionenwerte deuten auf Liquiditäts-Spitzen hin.
        </div>
    </div>
</body>
</html>
"""

def clean_code(raw):
    """Extrahiert die reine Zahl aus dem String."""
    s = str(raw).upper()
    # Entferne bekannte Rausch-Wörter
    s = re.sub(r'SEKTOR|KJPC2-|NACEIDX-|PCM2-|TIIDX-|IDX-', '', s)
    # Entferne alles, was kein Buchstabe oder Zahl ist
    s = s.strip().split(' ')[-1] # Nimm den letzten Teil, falls Leerzeichen da sind
    return s

def clean_val(raw):
    """Reinigt Zahlenformate (1.000,50 oder 100,50)."""
    if pd.isna(raw): return 0.0
    s = str(raw).replace(' EUR', '').replace('.', '').replace(',', '.')
    try:
        return float(s)
    except:
        return 0.0

def process():
    error = None
    try:
        # Lade die CSV
        df = pd.read_csv('data.csv', sep=None, engine='python', header=None)
        
        results = []
        for _, row in df.iterrows():
            best_code = None
            best_val = 0.0
            raw_code_display = "N/A"
            
            # Scanne jede Spalte in der Reihe
            for cell in row:
                cell_str = str(cell)
                # Suche nach einem Code (NACE, KJPC, etc.)
                if any(x in cell_str for x in ["NACE", "KJPC", "PCM", "47", "35", "10"]):
                    clean = clean_code(cell_str)
                    if clean in INDUSTRY_MAP or (clean.isdigit() and len(clean) in [2, 4]):
                        best_code = clean
                        raw_code_display = cell_str
                
                # Suche nach einem Wert (enthält Ziffern und Komma/Punkt)
                if any(char.isdigit() for char in cell_str):
                    val = clean_val(cell_str)
                    if val > best_val:
                        best_val = val
            
            if best_code:
                results.append({
                    "raw_code": raw_code_display,
                    "code": best_code,
                    "name": INDUSTRY_MAP.get(best_code, "Spezialsortiment"),
                    "val": best_val
                })
        
        if not results:
            error = "Keine passenden Branchen-Codes in der CSV gefunden."
            return [], error

        # Sortieren nach Wert
        results = sorted(results, key=lambda x: x['val'], reverse=True)
        return results, error
    except FileNotFoundError:
        return [], "Datei 'data.csv' nicht gefunden."
    except Exception as e:
        return [], str(e)

@app.route('/')
def index():
    data, error = process()
    # Falls gar nichts gefunden wurde, Fallback auf Demo zur Ansicht
    if not data and not error:
        data = [
            {"raw_code": "KJPC2-35", "code": "35", "name": "Energieversorgung", "val": 43983933.0},
            {"raw_code": "NACE-4791", "code": "4791", "name": "Online-Handel", "val": 103.4}
        ]
    return render_template_string(HTML_TEMPLATE, style=STYLE, data=data, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)