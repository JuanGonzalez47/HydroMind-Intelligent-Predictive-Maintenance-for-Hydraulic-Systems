import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import base64, io, joblib, warnings
from pathlib import Path
import numpy as np
from scipy.stats import skew, kurtosis
from numpy.polynomial import Polynomial

warnings.filterwarnings("ignore")

BASE_DIR   = Path.cwd()
MODELS_DIR = BASE_DIR / "models"

scaler = joblib.load(MODELS_DIR / "standard_scaler.pkl")

MODEL_CONFIGS = {
    "stable_flag": {
        "model":         joblib.load(MODELS_DIR / "stable_flag_model.pkl"),
        "features":      joblib.load(MODELS_DIR / "stable_flag_features.pkl"),
        "label_encoder": joblib.load(MODELS_DIR / "stable_flag_label_encoder.pkl"),
    },
    "cooler_condition": {
        "model":         joblib.load(MODELS_DIR / "cooler_condition_model.pkl"),
        "features":      joblib.load(MODELS_DIR / "cooler_condition_features.pkl"),
        "label_encoder": None,
    },
    "valve_condition": {
        "model":         joblib.load(MODELS_DIR / "valve_condition_model.pkl"),
        "features":      joblib.load(MODELS_DIR / "valve_condition_features.pkl"),
        "label_encoder": joblib.load(MODELS_DIR / "valve_condition_label_encoder.pkl"),
    },
    "internal_pump_leakage": {
        "model":         joblib.load(MODELS_DIR / "internal_pump_leakage_model.pkl"),
        "features":      joblib.load(MODELS_DIR / "internal_pump_leakage_features.pkl"),
        "label_encoder": joblib.load(MODELS_DIR / "internal_pump_leakage_label_encoder.pkl"),
    },
    "hydraulic_accumulator": {
        "model":         joblib.load(MODELS_DIR / "hydraulic_accumulator_model.pkl"),
        "features":      joblib.load(MODELS_DIR / "hydraulic_accumulator_features.pkl"),
        "label_encoder": joblib.load(MODELS_DIR / "hydraulic_accumulator_label_encoder.pkl"),
    },
}

SENSOR_FREQ = {
    "PS1":100,"PS2":100,"PS3":100,"PS4":100,"PS5":100,"PS6":100,"EPS1":100,
    "FS1":10,"FS2":10,
    "TS1":1,"TS2":1,"TS3":1,"TS4":1,"VS1":1,"CE":1,"CP":1,"SE":1,
}

WINDOW_SECONDS = 45
CYCLE_S        = WINDOW_SECONDS
TICK_MS        = CYCLE_S * 1000

SERVER_DATA = {}


# ── Inferencia ────────────────────────────────────────────────────────────────
def calc_slope(x):
    idx = np.arange(len(x))
    if len(x) < 2: return 0.0
    coef = Polynomial.fit(idx, x, 1).convert().coef
    return coef[1] if len(coef) >= 2 else 0.0

STATS = {"mean": np.mean, "std": np.std, "skew": skew,
         "kurtosis": kurtosis, "slope": calc_slope, "max": np.max}

def extract_features(row_dict):
    rec = {}
    for sensor, arr in row_dict.items():
        freq     = SENSOR_FREQ.get(sensor, 1)
        n_window = WINDOW_SECONDS * freq
        arr      = arr[:n_window] if len(arr) > n_window else arr
        for sname, fn in STATS.items():
            rec[f"{sensor}_{sname}"] = fn(arr)
    return pd.DataFrame([rec])

def run_model(target, X_raw):
    cfg           = MODEL_CONFIGS[target]
    feature_names = cfg["features"]
    X_c = pd.DataFrame(np.zeros((1, len(scaler.feature_names_in_))),
                       columns=scaler.feature_names_in_)
    for col in feature_names:
        if col in X_raw.columns:
            X_c[col] = X_raw[col].values
    X_s  = pd.DataFrame(scaler.transform(X_c), columns=scaler.feature_names_in_)
    pred = cfg["model"].predict(X_s[feature_names])[0]
    if cfg["label_encoder"] is not None:
        pred = cfg["label_encoder"].inverse_transform([int(pred)])[0]
    return pred

def predict_cycle(row_dict):
    X_raw       = extract_features(row_dict)
    stable_pred = run_model("stable_flag", X_raw)
    is_stable   = int(stable_pred) == 1
    result      = {"stable_flag": int(stable_pred), "is_stable": is_stable}
    for t in ["cooler_condition", "valve_condition",
              "internal_pump_leakage", "hydraulic_accumulator"]:
        result[t] = int(run_model(t, X_raw))
    return result


# ── Colores por componente ────────────────────────────────────────────────────
COOLER_LABELS  = {3:"Falla cercana (3%)", 20:"Eficiencia reducida (20%)", 100:"Eficiencia completa (100%)"}
VALVE_LABELS   = {100:"Óptimo (100%)", 90:"Falla pequeña (90%)", 80:"Falla severa (80%)", 73:"Falla total (73%)"}
PUMP_LABELS    = {0:"Sin fuga", 1:"Fuga débil", 2:"Fuga severa"}
ACCUM_LABELS   = {130:"Óptimo (130 bar)", 115:"Falla leve (115 bar)",
                  100:"Falla severa (100 bar)", 90:"Falla total (90 bar)"}

def cooler_color(val):
    if val is None: return "secondary"
    if val == 100:  return "success"
    if val == 20:   return "warning"
    return "danger"   # 3

def valve_color(val):
    if val is None: return "secondary"
    if val == 100:  return "success"
    if val == 90:   return "warning"
    if val == 80:   return "danger"
    return "danger"   # 73

def pump_color(val):
    if val is None: return "secondary"
    if val == 0:    return "success"
    if val == 1:    return "warning"
    return "danger"   # 2

def accum_color(val):
    if val is None: return "secondary"
    if val == 130:  return "success"
    if val == 115:  return "warning"
    if val == 100:  return "danger"
    return "danger"   # 90

def fmt(val, lmap):
    if val is None: return "—"
    return lmap.get(int(val), str(val))


# ── Semáforo general ──────────────────────────────────────────────────────────
def semaforo_general(diag):
    """
    Retorna (color, texto) para el badge principal del panel.
    """
    if diag is None:
        return "secondary", "Sin diagnóstico"

    cooler = diag.get("cooler_condition")
    valve  = diag.get("valve_condition")
    pump   = diag.get("internal_pump_leakage")
    accum  = diag.get("hydraulic_accumulator")

    # Condiciones críticas → rojo
    if pump == 2 or accum == 90:
        return "danger", "⛔ Paro inmediato recomendado"

    # Falla severa en cualquier componente → rojo
    if cooler == 3 or valve == 73 or accum == 100:
        return "danger", "🔴 Falla severa detectada"

    # Degradación → amarillo
    if cooler == 20 or valve in [80, 90] or pump == 1 or accum == 115:
        return "warning", "🟡 Degradación detectada"

    # Todo óptimo → verde
    return "success", "🟢 Sistema óptimo"


# ── Panel diagnóstico ─────────────────────────────────────────────────────────
def make_diagnosis_panel(diag, cycle_num=0):

    # ── Stable flag (arriba, donde antes decía "Estado del sistema") ──────────
    if diag is None:
        stable_badge = dbc.Badge(
            "Sin datos de señales", color="secondary",
            className="w-100 mb-1"
        )
    elif diag["stable_flag"] != 1:
        stable_badge = dbc.Badge(
            "✓ Señales estables", color="success",
            className="w-100 mb-1"
        )
    else:
        stable_badge = dbc.Badge(
            "⚠ Señales inestables", color="warning",
            className="w-100 mb-1"
        )

    # ── Semáforo general (donde antes estaba el alert de fallas) ─────────────
    sem_color, sem_txt = semaforo_general(diag)
    semaforo_badge = dbc.Badge(
        sem_txt, color=sem_color,
        className="fs-6 w-100 mb-2"
    )

    # ── Advertencia si señales inestables ─────────────────────────────────────
    advertencia = None
    if diag is not None and diag["stable_flag"] == 1:
        advertencia = dbc.Alert(
            "Predicción NO garantizada: señales inestables. "
            "Puede deberse a ruido, transición o fallo de sensor.",
            color="warning", className="mb-2 p-2 small"
        )

    # ── Fila de cada componente ───────────────────────────────────────────────
    def crow(icon, label, val, lmap, color_fn):
        txt   = fmt(val, lmap) if diag is not None else "—"
        color = color_fn(val)  if diag is not None else "secondary"
        return html.Div([
            html.Span(f"{icon}  {label}",
                      className="small text-muted d-block mb-1"),
            dbc.Badge(txt, color=color, className="w-100 text-wrap mb-3"),
        ])

    ctxt = (f"Último ciclo: #{cycle_num}"
            if cycle_num > 0 else "Sin ciclos completados aún")

    body = [
        # stable flag
        html.P("Estabilidad de señales",
               className="small text-muted mb-1 fw-bold"),
        stable_badge,
        html.Hr(className="my-2"),
        # semáforo general
        html.P("Estado del sistema",
               className="small text-muted mb-1 fw-bold"),
        semaforo_badge,
    ]

    if advertencia:
        body.append(advertencia)

    body += [
        html.Hr(className="my-2"),
        html.P("Componentes", className="small text-muted mb-2 fw-bold"),
        crow("🌡", "Cooler",
             diag["cooler_condition"]      if diag else None,
             COOLER_LABELS, cooler_color),
        crow("🔧", "Válvula",
             diag["valve_condition"]       if diag else None,
             VALVE_LABELS,  valve_color),
        crow("💧", "Bomba",
             diag["internal_pump_leakage"] if diag else None,
             PUMP_LABELS,   pump_color),
        crow("🔋", "Acumulador",
             diag["hydraulic_accumulator"] if diag else None,
             ACCUM_LABELS,  accum_color),
        html.Hr(className="my-2"),
        html.Small(ctxt, className="text-muted"),
    ]

    return dbc.Card([
        dbc.CardHeader(html.B("🩺 Diagnóstico en tiempo real")),
        dbc.CardBody(body)
    ], color="dark", outline=True)


# ── Figura (ciclo completo, sin streaming) ────────────────────────────────────
def make_figure(key, cycle_idx):
    if key not in SERVER_DATA:
        return go.Figure().update_layout(template='plotly_dark')

    arr      = SERVER_DATA[key]
    cycle_idx = cycle_idx % arr.shape[0]
    freq     = SENSOR_FREQ.get(key, 1)

    y = arr[cycle_idx, :].tolist()
    x = [i / freq for i in range(len(y))]

    fig = go.Figure(go.Scatter(
        x=x, y=y, mode='lines',
        line=dict(color='#00d4ff', width=1.2)
    ))
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(range=[0, CYCLE_S], title='Tiempo (s)', fixedrange=True),
        yaxis=dict(title=key, fixedrange=True),
        margin=dict(l=45, r=10, t=10, b=35),
        showlegend=False,
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.server.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

INIT_STORE = {"running": False, "diag": None, "cycle_num": 0}

app.layout = dbc.Container([

    html.H4("Hydraulic Sensor Analysis", className="mt-3 mb-0"),
    html.Small(
        f"Analizando ciclos completos de {CYCLE_S}s. "
        f"Actualización cada {TICK_MS/1000:.0f}s.",
        className="text-muted"
    ),
    html.Hr(),

    html.Div(id="alert-container"),

    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag o ', html.A('selecciona .txt')]),
                style={
                    'width':'100%','height':'60px','lineHeight':'60px',
                    'borderWidth':'1px','borderStyle':'dashed',
                    'borderRadius':'5px','textAlign':'center'
                },
                multiple=True
            ),
            html.Br(),
            dbc.Button("Iniciar Procesamiento", id='start-btn',
                       color='success', className="w-100 mb-2"),
            dbc.Button("Detener", id='stop-btn',
                       color='danger', className="w-100 mb-2"),
            html.Hr(),
            html.P("Sensores cargados:", className="small text-muted mb-1"),
            html.Div(id='sensor-list'),
        ], width=2),

        dbc.Col([
            # placeholder vacío hasta el primer tick
            html.Div(id='graphs-container'),
        ], width=7),

        dbc.Col([
            html.Div(id='diagnosis-panel',
                     children=make_diagnosis_panel(None, 0)),
        ], width=3),
    ]),

    dcc.Interval(id='interval-component', interval=TICK_MS,
                 n_intervals=0, disabled=True),
    dcc.Store(id='master-store', data=INIT_STORE),

], fluid=True)


# ── Callback: carga archivos ──────────────────────────────────────────────────
# Solo registra los datos en SERVER_DATA y muestra badges.
# NO genera gráficas todavía.
@app.callback(
    Output('sensor-list',  'children'),
    Output('master-store', 'data', allow_duplicate=True),
    Input('upload-data',   'contents'),
    State('upload-data',   'filename'),
    State('master-store',  'data'),
    prevent_initial_call=True
)
def load_files(contents, filenames, master):
    if not contents:
        raise dash.exceptions.PreventUpdate

    for content, name in zip(contents, filenames):
        key = name.replace('.txt','').upper()
        if key == "PROFILE":
            continue
        try:
            _, data_str = content.split(',', 1)
            df = pd.read_csv(
                io.BytesIO(base64.b64decode(data_str)),
                sep='\t', header=None, dtype='float32'
            )
            SERVER_DATA[key] = df.values
        except Exception as e:
            print(f"Error loading {name}: {e}")

    badges = [dbc.Badge(k, color="info", className="me-1 mb-1")
              for k in SERVER_DATA]

    return badges, master


# ── Callback: botones ─────────────────────────────────────────────────────────
@app.callback(
    Output('interval-component', 'disabled'),
    Output('master-store',       'data',    allow_duplicate=True),
    Input('start-btn',           'n_clicks'),
    Input('stop-btn',            'n_clicks'),
    State('master-store',        'data'),
    prevent_initial_call=True
)
def control(start, stop, master):
    if ctx.triggered_id == 'start-btn':
        master["running"]   = True
        master["cycle_num"] = 0   # reset al iniciar
        return False, master

    master["running"]   = False
    master["diag"]      = None
    master["cycle_num"] = 0
    return True, master


# ── Callback: tick ────────────────────────────────────────────────────────────
# Las gráficas se crean aquí por primera vez en el tick 1.
@app.callback(
    Output('graphs-container', 'children'),
    Output('master-store',     'data',     allow_duplicate=True),
    Output('diagnosis-panel',  'children'),
    Output('alert-container',  'children'),
    Input('interval-component', 'n_intervals'),
    State('master-store',       'data'),
    prevent_initial_call=True
)
def on_tick(n, master):
    if not master.get("running", False) or not SERVER_DATA:
        raise dash.exceptions.PreventUpdate

    c = master.get("cycle_num", 0)

    # recoger fila completa de cada sensor
    completed_rows = {}
    for key in SERVER_DATA:
        arr = SERVER_DATA[key]
        idx = c % arr.shape[0]
        completed_rows[key] = arr[idx, :].astype(float)

    # generar figuras con el ciclo actual
    graphs = [
        dbc.Card([
            dbc.CardHeader(html.B(
                f"{key}  —  {SENSOR_FREQ.get(key,1)} Hz  "
                f"({SERVER_DATA[key].shape[1]} muestras/ciclo)"
            )),
            dbc.CardBody(dcc.Graph(
                id={'type':'sensor-graph','index':key},
                style={'height':'200px'},
                config={'displayModeBar': False},
                figure=make_figure(key, c)
            ))
        ], className="mb-2", color="dark", outline=True)
        for key in SERVER_DATA
    ]

    # inferencia
    try:
        diag = predict_cycle(completed_rows)
        master["diag"] = diag
        print(f"Ciclo #{c+1} → stable={diag['stable_flag']} | "
              f"cooler={diag['cooler_condition']} | "
              f"valve={diag['valve_condition']} | "
              f"pump={diag['internal_pump_leakage']} | "
              f"accum={diag['hydraulic_accumulator']}")
    except Exception as e:
        print(f"Inference error: {e}")
        diag = master.get("diag")

    master["cycle_num"] = c + 1

    return (
        graphs,
        master,
        make_diagnosis_panel(diag, c + 1),
        "",   # alert eliminado, semáforo ya está en el panel
    )


if __name__ == '__main__':
    app.run(debug=False)