from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configure matplotlib
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# Load model (try several common paths)
model = None
feature_cols = []
model_paths = [
    "flight_delay_model.pkl",
    "Model/flight_delay_model.pkl",
    "model/flight_delay_model.pkl",
]
feature_paths = [
    "model_features.pkl",
    "Model/model_features.pkl",
    "model/model_features.pkl",
]
loaded_model_path = None
loaded_feature_path = None
for p in model_paths:
    try:
        model = joblib.load(p)
        loaded_model_path = p
        break
    except Exception:
        model = None

for p in feature_paths:
    try:
        feature_cols = joblib.load(p)
        loaded_feature_path = p
        break
    except Exception:
        feature_cols = []

if model is None or not feature_cols:
    import traceback
    print("Warning: Model or feature list not loaded. Paths tried:", model_paths, feature_paths)
    traceback.print_stack()
else:
    print(f"Loaded model from: {loaded_model_path} and features from: {loaded_feature_path}")

THRESHOLD = 0.45
AIRLINES = ['AS', 'B6', 'DL', 'EV', 'F9', 'HA', 'MQ', 'NK', 'OO', 'UA', 'US', 'VX', 'WN']
AIRPORTS = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'JFK', 'LAS', 'SFO', 'SEA', 'MIA']
MONTHS = list(range(1, 13))
DAYS = list(range(1, 32))
DAY_OF_WEEKS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DEP_BLOCKS = ['00-05', '06-11', '12-17', '18-23']

def generate_chart_image(chart_type='gauge', proba=0):
    """Generate chart as base64 image"""
    if chart_type == 'gauge':
        fig, ax = plt.subplots(figsize=(10, 2.5), facecolor='white')
        risk_color = '#ef4444' if proba >= THRESHOLD else '#eab308' if proba >= 0.30 else '#22c55e'
        
        ax.barh(['Delay Risk'], [proba*100], height=0.6, color=risk_color, alpha=0.85, edgecolor='white', linewidth=2)
        ax.barh(['Delay Risk'], [100-proba*100], left=[proba*100], height=0.6, color='#e5e7eb', alpha=0.4)
        ax.axvline(THRESHOLD*100, color='#1e40af', linestyle='--', linewidth=2.5, label=f'High Risk Threshold ({THRESHOLD*100:.0f}%)', alpha=0.8)
        
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", fontsize=11, fontweight='600', color='#334155')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        
    elif chart_type == 'stats':
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        stats_data = pd.DataFrame({
            'Metric': ['Delayed', 'On-Time', 'Cancelled'],
            'Percentage': [22, 76, 2]
        })
        colors = ['#ef4444', '#22c55e', '#fbbf24']
        wedges, texts, autotexts = ax.pie(stats_data['Percentage'], labels=stats_data['Metric'], autopct='%1.1f%%', 
                   colors=colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title("Flight Status Distribution", fontweight='bold', fontsize=13, color='#1e293b', pad=20)
        
    elif chart_type == 'hourly':
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        hours = np.arange(0, 24)
        delays = np.sin(hours/4) * 10 + 20 + np.random.normal(0, 2, 24)
        
        ax.plot(hours, delays, marker='o', linewidth=3, color='#1e40af', markersize=7, markerfacecolor='#3b82f6', markeredgewidth=2, markeredgecolor='white')
        ax.fill_between(hours, delays, alpha=0.15, color='#1e40af')
        ax.set_xlabel("Hour of Day", fontsize=11, fontweight='600', color='#334155')
        ax.set_ylabel("Avg Delay (min)", fontsize=11, fontweight='600', color='#334155')
        ax.set_title("Average Delay by Departure Hour", fontweight='bold', fontsize=13, color='#1e293b', pad=20)
        ax.grid(alpha=0.2, linestyle='--')
        ax.set_facecolor('white')
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         airlines=AIRLINES,
                         airports=AIRPORTS,
                         months=MONTHS,
                         days=DAYS,
                         day_of_weeks=DAY_OF_WEEKS,
                         dep_blocks=DEP_BLOCKS)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        data = request.json
        # If model or feature list failed to load at startup, return a clear JSON error
        if model is None or not feature_cols:
            return jsonify({
                'success': False,
                'error': 'Model or feature list not loaded on server. Check server logs for details.'
            }), 503
        
        # Extract all features from request
        # Support either a combined 'date' (YYYY-MM-DD) or separate month/day values
        date_str = data.get('date')
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                month = dt.month
                day = dt.day
                # weekday(): Monday is 0 -> convert to 1..7 if needed elsewhere
                day_of_week = dt.weekday() + 1
            except Exception:
                # fall back to defaults if parsing fails
                month = int(data.get('month', 1))
                day = int(data.get('day', 1))
                day_of_week = int(data.get('day_of_week', 0))
        else:
            month = int(data.get('month', 1))
            day = int(data.get('day', 1))
            day_of_week = int(data.get('day_of_week', 0))

        origin_airport = data.get('origin_airport', 'ATL')
        destination_airport = data.get('destination_airport', 'LAX')
        # Accept scheduled_arrival as either numeric HHMM (e.g. 1200) or string 'HH:MM' (e.g. '12:00')
        raw_sched = data.get('scheduled_arrival', 1200)
        try:
            if isinstance(raw_sched, str) and ':' in raw_sched:
                parts = raw_sched.split(':')
                hh = int(parts[0])
                mm = int(parts[1])
                if not (0 <= hh <= 23 and 0 <= mm <= 59):
                    return jsonify({'success': False, 'error': 'scheduled_arrival out of range (00:00-23:59)'}), 400
                scheduled_arrival = hh * 100 + mm
            else:
                scheduled_arrival = int(raw_sched)
            # final sanity check: minutes part < 60 and within 0..2359
            if scheduled_arrival < 0 or scheduled_arrival > 2359 or (scheduled_arrival % 100) >= 60:
                return jsonify({'success': False, 'error': 'scheduled_arrival invalid. Use HH:MM or HHMM within 00:00-23:59'}), 400
        except Exception:
            return jsonify({'success': False, 'error': 'Invalid scheduled_arrival format (use HH:MM or HHMM)'}), 400
        distance = float(data.get('distance', 1000))
        dep_hour = int(data.get('dep_hour', 10))
        airline = data.get('airline', 'DL')
        dep_block = data.get('dep_block', '06-11')
        
        # Build input dictionary
        input_dict = {col: 0 for col in feature_cols}
        
        # Set numerical features
        input_dict['MONTH'] = month
        input_dict['DAY'] = day
        input_dict['DAY_OF_WEEK'] = day_of_week
        input_dict['ORIGIN_AIRPORT'] = ord(origin_airport[0]) - ord('A')  # Simple encoding
        input_dict['DESTINATION_AIRPORT'] = ord(destination_airport[0]) - ord('A')
        input_dict['SCHEDULED_ARRIVAL'] = scheduled_arrival
        input_dict['DISTANCE'] = distance
        input_dict['DEP_HOUR'] = dep_hour
        
        # Set airline features
        airline_key = f"AIRLINE_{airline}"
        if airline_key in feature_cols:
            input_dict[airline_key] = 1
        
        # Set departure block features
        if dep_block == '06-11':
            input_dict['DEP_BLOCK_6-11'] = 1
        elif dep_block == '12-17':
            input_dict['DEP_BLOCK_12-17'] = 1
        elif dep_block == '18-23':
            input_dict['DEP_BLOCK_18-23'] = 1
        
        X_input = pd.DataFrame([input_dict], columns=feature_cols)
        proba = float(model.predict_proba(X_input)[0][1])
        
        # Determine risk level
        if proba >= THRESHOLD:
            risk_level = 'HIGH'
            risk_color = '#dc2626'
            emoji = '🔴'
            recommendations = [
                "Consider booking an earlier flight",
                "Allow extra travel time to the airport",
                "Check airport and airline alerts regularly"
            ]
        elif proba >= 0.30:
            risk_level = 'MODERATE'
            risk_color = '#d97706'
            emoji = '🟡'
            recommendations = ["Monitor flight status before departure"]
        else:
            risk_level = 'LOW'
            risk_color = '#16a34a'
            emoji = '🟢'
            recommendations = ["Flight shows excellent on-time probability. Safe to proceed!"]
        
        # Generate gauge chart
        gauge_image = generate_chart_image('gauge', proba)
        
        return jsonify({
            'success': True,
            'proba': proba,
            'proba_percent': f"{proba*100:.1f}%",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'emoji': emoji,
            'recommendations': recommendations,
            'gauge_image': gauge_image
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/analytics')
def analytics():
    """API endpoint for analytics charts"""
    try:
        stats_image = generate_chart_image('stats')
        hourly_image = generate_chart_image('hourly')
        
        return jsonify({
            'success': True,
            'stats_image': stats_image,
            'hourly_image': hourly_image
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/status')
def status():
    """Return server/model loading status for UI"""
    try:
        return jsonify({
            'success': True,
            'model_loaded': model is not None,
            'features_loaded': bool(feature_cols),
            'loaded_model_path': loaded_model_path,
            'loaded_feature_path': loaded_feature_path
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
