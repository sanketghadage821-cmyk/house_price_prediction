import pickle
from flask import Flask, render_template_string, request
import pandas as pd

app = Flask(__name__)

# Load the trained linear regression model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Exact feature names extracted from linear_Model.pkl
FEATURE_NAMES = [
    'number of bedrooms',
    'number of bathrooms',
    'living area',
    'lot area',
    'number of floors',
    'waterfront present',
    'number of views',
    'condition of the house',
    'grade of the house',
    'Area of the house(excluding basement)',
    'Area of the basement',
    'Built Year',
    'Renovation Year',
    'lot_area_renov',
    'Number of schools nearby',
    'Distance from the airport'
]

# Embedded UI Template (Tailwind CSS with Slate & Indigo Theme)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        }
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .input-field {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #f8fafc;
            transition: all 0.2s ease-in-out;
        }
        .input-field:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.3);
        }
    </style>
</head>
<body class="min-h-screen text-slate-100 flex items-center justify-center p-4 sm:p-8">

    <div class="max-w-5xl w-full">
        <div class="text-center mb-8">
            <span class="inline-block px-3 py-1 bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 rounded-full text-xs font-semibold uppercase tracking-wider mb-3">
                Machine Learning Model
            </span>
            <h1 class="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
                Real Estate Valuation Tool
            </h1>
            <p class="text-slate-400 text-sm sm:text-base mt-2">
                Enter property parameters below to compute estimated market value.
            </p>
        </div>

        {% if prediction_text %}
        <div class="mb-8 p-6 glass-card rounded-2xl border-indigo-500/40 text-center shadow-xl">
            <p class="text-sm uppercase tracking-wider text-slate-400 font-semibold mb-1">Estimated Valuation</p>
            <h2 class="text-4xl sm:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-200">
                {{ prediction_text }}
            </h2>
        </div>
        {% endif %}

        {% if error_text %}
        <div class="mb-8 p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-center text-sm">
            {{ error_text }}
        </div>
        {% endif %}

        <form action="/predict" method="POST" class="glass-card rounded-3xl p-6 sm:p-10 shadow-2xl space-y-8">
            <div>
                <h3 class="text-lg font-bold text-indigo-300 mb-4 border-b border-slate-700/60 pb-2">
                    Room Layout & Dimensions
                </h3>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Bedrooms</label>
                        <input type="number" step="any" name="number of bedrooms" required value="3" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Bathrooms</label>
                        <input type="number" step="any" name="number of bathrooms" required value="2" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Number of Floors</label>
                        <input type="number" step="any" name="number of floors" required value="1" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Living Area (sq ft)</label>
                        <input type="number" step="any" name="living area" required value="2000" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Lot Area (sq ft)</label>
                        <input type="number" step="any" name="lot area" required value="5000" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Renovated Lot Area</label>
                        <input type="number" step="any" name="lot_area_renov" required value="5000" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                </div>
            </div>

            <div>
                <h3 class="text-lg font-bold text-indigo-300 mb-4 border-b border-slate-700/60 pb-2">
                    Area & Structure Details
                </h3>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Area Excl. Basement</label>
                        <input type="number" step="any" name="Area of the house(excluding basement)" required value="1500" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Basement Area</label>
                        <input type="number" step="any" name="Area of the basement" required value="500" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Built Year</label>
                        <input type="number" name="Built Year" required value="1995" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Renovation Year</label>
                        <input type="number" name="Renovation Year" required value="0" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">House Condition (1-5)</label>
                        <input type="number" step="any" name="condition of the house" required value="3" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Grade of House (1-13)</label>
                        <input type="number" step="any" name="grade of the house" required value="7" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                </div>
            </div>

            <div>
                <h3 class="text-lg font-bold text-indigo-300 mb-4 border-b border-slate-700/60 pb-2">
                    Location & View
                </h3>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Waterfront Present</label>
                        <select name="waterfront present" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                            <option value="0" selected>No (0)</option>
                            <option value="1">Yes (1)</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Number of Views</label>
                        <input type="number" step="any" name="number of views" required value="0" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Schools Nearby</label>
                        <input type="number" step="any" name="Number of schools nearby" required value="2" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">Airport Dist. (km)</label>
                        <input type="number" step="any" name="Distance from the airport" required value="15" class="input-field w-full px-4 py-2.5 rounded-xl text-sm">
                    </div>
                </div>
            </div>

            <div class="pt-4">
                <button type="submit" class="w-full py-4 px-8 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:from-indigo-600 hover:via-purple-600 hover:to-pink-600 text-white font-bold rounded-2xl shadow-lg shadow-indigo-500/25 transition-all duration-200 text-base">
                    Calculate Valuation
                </button>
            </div>
        </form>
    </div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_values = [float(request.form.get(feature, 0)) for feature in FEATURE_NAMES]
        input_data = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
        prediction = model.predict(input_data)[0]
        formatted_price = f"${max(0, prediction):,.2f}"
        return render_template_string(HTML_TEMPLATE, prediction_text=formatted_price)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
