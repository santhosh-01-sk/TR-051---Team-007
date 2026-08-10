# DisasterAI — AI-Powered Satellite Disaster Damage Assessment

![DisasterAI Platform](https://img.shields.io/badge/Platform-DisasterAI-4F46E5?style=for-the-badge&logo=satellite)
![Python](https://img.shields.io/badge/Backend-Python%20%7C%20Flask-3776AB?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/ML-PyTorch%20%7C%20Siamese%20U--Net-EE4C2C?style=for-the-badge&logo=pytorch)
![Frontend](https://img.shields.io/badge/Frontend-HTML5%20%7C%20CSS3%20%7C%20React-61DAFB?style=for-the-badge&logo=react)

**DisasterAI** is an automated disaster damage assessment platform leveraging **Siamese Neural Network-based Change Detection** to analyze before-and-after satellite imagery. The platform detects structural anomalies, classifies damage severity into standard categories, computes affected spatial statistics, prioritizes emergency aid deployment zones, and exports geospatial data in GeoJSON format.

---

## 🌟 Key Features

- **Siamese U-Net Change Detection**: Analyzes spectral signatures and geometry differences between pre-disaster and post-disaster satellite imagery.
- **Dual Assessment Modes**:
  - **Manual Imagery Upload**: Upload custom pre/post satellite pairs (supports PNG, JPG, WEBP, TIFF) with instant file dimension and format extraction.
  - **Sentinel-2 Satellite Search**: Select an Area of Interest (AOI) on an interactive map to automatically retrieve Sentinel-2 L2A satellite imagery.
- **Interactive Damage Severity Map**:
  - Pre vs Post image swipe slider with color-coded damage segmentation overlays.
  - Distribution breakdown across four damage classes:
    - 🟢 **No Damage**
    - 🟡 **Minor Damage**
    - 🟠 **Major Damage**
    - 🔴 **Destroyed**
- **Aid Deployment Priority Ranking**: Ranks geographic sectors into **Critical (Zone A)**, **High (Zone B)**, **Medium (Zone C)**, and **Low (Zone D)** priority zones to streamline disaster response dispatch.
- **GeoJSON Spatial Data Export**: Download complete `FeatureCollection` geospatial layers compatible with QGIS, ArcGIS, and emergency dispatch systems.
- **Stepped Pipeline Progress State**: 5-stage animated progress feedback (*Analyzing Satellite Images...* → *Detecting Changes...* → *Classifying Damage...* → *Generating Damage Map...* → *Preparing Assessment Results...*).
- **Honest Demo / Live Model Labeling**: Transparently indicates when operating in **Live AI Assessment** vs **Demo Assessment** mode.

---

## 📁 Repository Structure

```
Disaster/
├── backend/
│   ├── app.py                # Flask Web Server & API Endpoints (/ingest, /auto_assess)
│   ├── model.py              # Siamese U-Net Neural Network Architecture (PyTorch)
│   ├── pipeline.py           # Disaster Assessment Inference Pipeline
│   ├── geo_processor.py      # Spatial Mask to GeoJSON & Geographic Stats Calculator
│   ├── sentinel_fetcher.py   # Sentinel-2 Imagery Fetcher & API Integration
│   ├── test_run.py           # Pipeline Verification Script
│   └── requirements.txt      # Python Dependencies
├── frontend/
│   └── disaster.html         # DisasterAI Web Interface (HTML5, Glassmorphism CSS, React, Leaflet)
├── .gitignore                # Git Ignore Configurations
└── README.md                 # Project Documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.10 or higher
- **Node/Browser**: Any modern web browser (Chrome, Firefox, Edge, Safari)

### Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/santhosh-01-sk/TR-051---Team-007.git
   cd TR-051---Team-007
   ```

2. **Set up Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Launch the Backend Server**:
   ```bash
   python backend/app.py
   ```
   *The server will start at `http://127.0.0.1:8000`.*

5. **Open the Web Application**:
   Open `http://127.0.0.1:8000` in your web browser, or open `frontend/disaster.html` directly for client-side demo mode.

---

## 🛰️ API Endpoints

### `POST /ingest`
Submits uploaded pre and post disaster images along with the disaster category for AI inference.

- **Request Form Data**:
  - `pre_image`: Pre-disaster satellite image file
  - `post_image`: Post-disaster satellite image file
  - `disaster_type`: Disaster category ID (e.g. `earthquake`, `flood`, `wildfire`)
- **Response**: JSON object containing damage masks (`base64`), heatmap overlay (`base64`), impact stats ($\text{km}^2$, $\text{ha}$, percentages), reasoning formulation, and GeoJSON string.

### `POST /auto_assess`
Retrieves satellite imagery for a given Bounding Box (AOI) and date, then runs the assessment pipeline.

- **Request JSON**:
  ```json
  {
    "bbox": [west, south, east, north],
    "date": "YYYY-MM-DD",
    "disaster_type": "earthquake"
  }
  ```
- **Response**: JSON object containing retrieved pre/post base64 tiles, heatmap overlay, spatial statistics, and geo-referenced GeoJSON.

---

## 🛠️ Technology Stack

- **ML & Backend**: PyTorch, Python 3, Flask, Flask-CORS, NumPy, Pillow (PIL), GeoJSON
- **Frontend UI**: HTML5, Vanilla CSS3 (Glassmorphism, CSS Grid/Flexbox), JavaScript (ES6+), React 18, Chart.js, Leaflet.js
- **Typography & Aesthetics**: Google Outfit font, sleek dark navy (`#0F172A`) & emerald/indigo palette

---

## 📄 License

This project is open-source and available under the MIT License.
