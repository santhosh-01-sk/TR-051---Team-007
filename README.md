# DisasterAI — AI-Powered Satellite Disaster Damage Assessment

![DisasterAI Platform](https://img.shields.io/badge/Platform-DisasterAI-4F46E5?style=for-the-badge&logo=satellite)
![Vercel Deployment](https://img.shields.io/badge/Deployment-Vercel%20Ready-000000?style=for-the-badge&logo=vercel)
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
- **Honest Demo / Live Model Labeling**: Transparently indicates **Demo Assessment — AI model not trained** when operating in frontend demo mode.

---

## 📁 Repository Structure

```
Disaster/
├── index.html                # Vercel Static Frontend Entry Point (Root)
├── backend/                  # Python/Flask Machine Learning Backend
│   ├── app.py                # Flask Web Server & API Endpoints (/ingest, /auto_assess)
│   ├── model.py              # Siamese U-Net Neural Network Architecture (PyTorch)
│   ├── pipeline.py           # Disaster Assessment Inference Pipeline
│   ├── geo_processor.py      # Spatial Mask to GeoJSON & Geographic Stats Calculator
│   ├── sentinel_fetcher.py   # Sentinel-2 Imagery Fetcher & API Integration
│   ├── test_run.py           # Pipeline Verification Script
│   └── requirements.txt      # Python Dependencies
├── frontend/
│   └── disaster.html         # Frontend Source Copy
├── .gitignore                # Git Ignore Configurations
└── README.md                 # Project Documentation
```

---

## 🌐 Deploying Frontend to Vercel

The frontend entry point `index.html` is located at the root of the repository, making it 100% compatible with instant Vercel static deployment without requiring Python, Flask, or backend server configuration.

### Deployment Steps:

#### Option A: Via GitHub Integration (Recommended)
1. Push your repository to GitHub.
2. Log in to [Vercel Dashboard](https://vercel.com/dashboard).
3. Click **"Add New..."** → **"Project"**.
4. Import your GitHub repository (`TR-051---Team-007`).
5. Keep **Framework Preset** as **"Other"** (or Static HTML).
6. Click **Deploy**. Vercel will instantly host `index.html` live.

#### Option B: Via Vercel CLI
```bash
npm install -g vercel
vercel login
vercel --prod
```

> **Note on Demo Mode**: The Vercel deployment runs entirely client-side in the browser. It does not attempt calls to `127.0.0.1:8000` or `localhost:8000`, ensuring fast and reliable demo rendering without server requirements.

---

## 🚀 Local Development

### Running Frontend Locally
Simply open `index.html` in any web browser, or serve it using Python:
```bash
python -m http.server 8080
```
Open `http://localhost:8080` in your browser.

### Running Backend Server (Local AI Inference)

1. **Install Python Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Launch Flask Backend**:
   ```bash
   python backend/app.py
   ```
   *The backend will listen on `http://127.0.0.1:8000`.*

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
