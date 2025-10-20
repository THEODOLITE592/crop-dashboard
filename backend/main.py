# fastapi_ndvi_etc_api.py
from fastapi import FastAPI, HTTPException , UploadFile, File , Body
from pydantic import BaseModel
from typing import Optional ,  Dict, Any , List
import ee, math, datetime, requests
from google.oauth2 import service_account
import tensorflow as tf
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import base64
import io
import os
import json
from fastapi.staticfiles import StaticFiles
from pathlib import Path


# ---------------- Paths Setup ----------------
BASE_DIR = os.path.dirname(__file__)  # backend folder

# .env file in backend/
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

# Read environment variables
api_key = os.getenv("GOOGLE_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")
if not PROJECT_ID:
    raise ValueError("PROJECT_ID not found in .env")

# Configure Google Generative AI
genai.configure(api_key=api_key)

# Service account JSON inside backend/key/
service_account_path = os.path.join(BASE_DIR, "key", "crop-472500-54a3e5f232ea.json")
if not os.path.exists(service_account_path):
    raise FileNotFoundError(f"Service account file not found: {service_account_path}")

# ---------------- Earth Engine Initialization ----------------
credentials = service_account.Credentials.from_service_account_file(
    service_account_path,
    scopes=[
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/cloud-platform"
    ]
)
ee.Initialize(credentials, project=PROJECT_ID)

# ---------------- FastAPI App ----------------
app = FastAPI(title="NDVI & ETc API")

# Path to frontend folder relative to backend
frontend_path = Path(__file__).parent.parent / "frontend"

# Serve frontend at root
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


# ---------------- Load Trained Model ----------------
model_path = os.path.join(BASE_DIR, "trained_final_model_2.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found: {model_path}")

model = tf.keras.models.load_model(model_path)

COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_THRESHOLD = 50
SCALE = 10

# ---------------- Crop info ----------------
kc_range = {
    "rice":      [(0.7, 1.0), (0.9, 1.2), (0.9, 1.2), (0.6, 0.9)],
    "maize":     [(0.3, 0.4), (0.5, 0.8), (0.9, 1.2), (0.4, 0.6)],
    "wheat":     [(0.2, 0.3), (0.4, 0.6), (0.9, 1.15), (0.2, 0.25)],
    "cotton":    [(0.2, 0.3), (0.5, 0.8), (0.9, 1.15), (0.8, 0.95)],
    "groundnut": [(0.3, 0.4), (0.5, 0.8), (0.8, 1.0), (0.5, 0.7)],
    "sugarcane": [(0.5, 0.7), (0.7, 0.9), (1.0, 1.2), (0.9, 1.05)]
}

crop_stage_days = {
    "rice":      [(0, 30), (31, 70), (71, 120), (121, 150)],
    "maize":     [(0, 25), (26, 60), (61, 90), (91, 120)],
    "wheat":     [(0, 20), (21, 50), (51, 90), (91, 120)],
    "cotton":    [(0, 30), (31, 80), (81, 150), (151, 200)],
    "groundnut": [(0, 25), (26, 60), (61, 110), (111, 140)],
    "sugarcane": [(0, 60), (61, 180), (181, 360), (361, 450)]
}

class_name = [
                'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
                'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
                'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
                'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
                'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
                'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
                'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
            ]

# ---------------- Request model ----------------
class NDVIRequest(BaseModel):
    coords: List[List[float]]
    crop: str
    sowing_date: str  # 'YYYY-MM-DD'
    end_date: Optional[str] = None
    visualcrossing_key: Optional[str] = None

# ---------------- Helper functions ----------------
def _to_date(obj):
    if isinstance(obj, datetime.date):
        return obj
    if isinstance(obj, str):
        return datetime.datetime.strptime(obj, "%Y-%m-%d").date()
    raise ValueError("Date must be datetime.date or 'YYYY-MM-DD' string")

def ra_daily(latitude_deg, day_of_year):
    phi = math.radians(latitude_deg)
    G_sc = 0.0820
    dr = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
    delta = 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)
    omega_s = math.acos(-math.tan(phi) * math.tan(delta))
    Ra = (24*60/math.pi) * G_sc * dr * (
        omega_s*math.sin(phi)*math.sin(delta) + math.cos(phi)*math.cos(delta)*math.sin(omega_s)
    )
    return Ra

def compute_et0_hargreaves(tmin, tmax, tavg, ra):
    return 0.0023 * (tavg + 17.8) * math.sqrt(tmax - tmin) * ra

def get_stage_for_day(crop, sowing_date, target_date):
    day_num = (target_date - sowing_date).days
    stages = crop_stage_days[crop]
    for idx, (start, end) in enumerate(stages):
        if start <= day_num <= end:
            return idx
    return len(stages) - 1

def get_kc_for_stage(crop, stage_idx):
    k_min, k_max = kc_range[crop][stage_idx]
    return (k_min + k_max) / 2

# 1) Use native Sentinel-2 scale
SCALE = 10   # native resolution for B4/B8 (Sentinel-2)

# 2) Optional SCL-based cloud mask for COPERNICUS/S2_SR_HARMONIZED
def mask_s2_sr_clouds(img):
    """
    Simple SCL-based mask: removes cloud shadow (3), clouds (8,9), thin cirrus (10), snow (11).
    Returns masked image.
    """
    scl = img.select('SCL')
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return img.updateMask(bad.Not())

# 3) Robust get_ndvi_list_from_sowing (polygon-only)
def get_ndvi_list_from_sowing(coords: List[List[float]], crop, sowing_date, end_date=None,
                              cloud_thresh=CLOUD_THRESHOLD, scale=SCALE, collection_id=COLLECTION_ID):
    """
    coords: list of [lon, lat] pairs forming outer ring. Returns sorted list of {'date','ndvi'}.
    Uses reduceColumns(...).get('list').getInfo() to safely fetch date/ndvi pairs.
    """
    if not coords or len(coords) < 3:
        raise ValueError("coords must be a list of at least 3 [lon,lat] points forming a polygon.")

    s_date = _to_date(sowing_date)
    e_date = _to_date(end_date) if end_date else datetime.date.today()

    field_geom = ee.Geometry.Polygon([coords])

    col = (ee.ImageCollection(collection_id)
           .filterBounds(field_geom)
           .filterDate(s_date.isoformat(), (e_date + datetime.timedelta(days=1)).isoformat())
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
           .map(lambda img: img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'))))

    # apply SCL mask if available
    try:
      pass  #  col = col.map(mask_s2_sr_clouds)
    except Exception:
        pass

    def _img_to_feat(img):
        date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
        mean_dict = img.select('NDVI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=field_geom,
            scale=scale,
            bestEffort=True,
            maxPixels=1e13
        )
        nd = mean_dict.get('NDVI')
        return ee.Feature(None, {'date': date, 'ndvi': nd})

    feats = col.map(_img_to_feat).filter(ee.Filter.notNull(['ndvi']))

    # safer client-side conversion to list-of-[date,ndvi]
    try:
        col_list = feats.reduceColumns(
            reducer=ee.Reducer.toList(2),
            selectors=['date', 'ndvi']
        ).get('list').getInfo()
    except Exception as ex:
        # If getInfo fails (too large or timeout), return empty and log
        print("Error fetching NDVI series (getInfo failed):", ex)
        col_list = []

    samples = []
    for item in col_list:
        if not item or len(item) < 2:
            continue
        dt, ndv = item[0], item[1]
        if dt and ndv is not None:
            try:
                samples.append({'date': dt, 'ndvi': round(float(ndv), 4)})
            except Exception:
                continue

    return sorted(samples, key=lambda x: x['date'])

GLOBAL_NDVI_ETC_STORAGE: Dict[str, Dict[str, Any]] = {}

# 4) Updated ndvi_etc endpoint (uses polygon centroid for weather + error handling)
@app.post("/ndvi_etc")
def ndvi_etc(req: NDVIRequest):
    if not req.visualcrossing_key:
        raise HTTPException(status_code=400, detail="Visual Crossing key missing")

    coords = req.coords

    # ------------------- New validation and area check -------------------
    if not coords or len(coords) < 3:
        raise HTTPException(status_code=400, detail="coords (polygon) must contain at least 3 [lon,lat] points.")

    # Quick sanity: ensure numeric [lon, lat]
    try:
        coords = [[float(c[0]), float(c[1])] for c in coords]
    except Exception:
        raise HTTPException(status_code=400, detail="coords must be numeric [lon,lat] pairs.")

    # Area check (limit e.g. 50 km²)
    field_geom = ee.Geometry.Polygon([coords])
    try:
        area_m2 = field_geom.area().getInfo()
        MAX_AREA = 50_000_000  # 50 km²
        if area_m2 > MAX_AREA:
            raise HTTPException(
                status_code=400,
                detail=f"Polygon area too large ({area_m2:.0f} m²). Please use smaller field polygons."
            )
    except Exception as ex:
        # If getInfo fails, proceed but log
        print("Area check getInfo failed:", ex)
    # -------------------------------------------------------------------

    try:
        ndvi_samples = get_ndvi_list_from_sowing(
            coords=coords,
            crop=req.crop,
            sowing_date=req.sowing_date,
            end_date=req.end_date
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NDVI extraction failed: {e}")

    if not ndvi_samples:
        return {
            "crop": req.crop,
            "date_range": [req.sowing_date, req.end_date if req.end_date else datetime.date.today().isoformat()],
            "ndvi_samples": [],
            "results": {},
            "message": "No NDVI samples found for polygon/date range (possible cloud cover or no images)."
        }

    # Compute centroid coords for weather API queries
    try:
        centroid_coords = ee.Geometry.Polygon([coords]).centroid().getInfo()['coordinates']
        weather_lon, weather_lat = centroid_coords[0], centroid_coords[1]
    except Exception as ex:
        print("Centroid extraction failed, falling back to first vertex:", ex)
        weather_lon, weather_lat = coords[0][0], coords[0][1]

    # compute ETc using centroid for weather queries
    try:
        results = compute_etc_from_ndvi_dates_with_safe_requests(
            ndvi_samples, weather_lat, weather_lon, req.crop, req.visualcrossing_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ETc computation failed: {e}")
    
    # ---------------- Store in global dictionary ----------------
    ndvi_samples_list = []
    for date_str, values in results.items():
        ndvi_samples_list.append({
            "Date": date_str,
            "NDVI": values.get("ndvi"),
            "Tmax": values.get("tmax"),
            "Tmin": values.get("tmin"),
            "Tavg": values.get("tavg"),
            "Kc": values.get("Kc"),
            "ETc": values.get("ETc")
        })

    GLOBAL_NDVI_ETC_STORAGE[req.crop] = {
        "date_range": [req.sowing_date, req.end_date if req.end_date else datetime.date.today().isoformat()],
        "ndvi_samples": ndvi_samples_list,
        "results": results
    }

    return {
        "crop": req.crop,
        "date_range": [req.sowing_date, req.end_date if req.end_date else datetime.date.today().isoformat()],
        "ndvi_samples": ndvi_samples,
        "results": results
    }

# 5) Small wrapper for safe VisualCrossing calls (replace compute_etc_from_ndvi_dates or add this wrapper)
def compute_etc_from_ndvi_dates_with_safe_requests(ndvi_list, lat, lon, crop, api_key):
    results = {}
    if not ndvi_list:
        return results
    sowing_date = datetime.datetime.strptime(ndvi_list[0]['date'], "%Y-%m-%d").date()
    for entry in ndvi_list:
        date_str = entry['date']
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date_str}/{date_str}?unitGroup=metric&include=days&key={api_key}&contentType=json"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if 'days' not in data or not data['days']:
                raise ValueError("No day data returned")
            day_data = data['days'][0]
            tmax = day_data.get('tempmax')
            tmin = day_data.get('tempmin')
            tavg = day_data.get('temp', (tmax + tmin) / 2 if tmax is not None and tmin is not None else None)
        except Exception as ex:
            print(f"Weather API failed for {date_str} at {lat},{lon}: {ex}")
            # Skip this date or supply Nones
            results[date_str] = {"ndvi": entry['ndvi'], "tmax": None, "tmin": None, "tavg": None, "Kc": None, "ETc": None}
            continue

        # ET0 + ETc
        ra = ra_daily(lat, date_obj.timetuple().tm_yday)
        # guard if tmax/tmin missing
        if tmax is None or tmin is None or tavg is None or tmax - tmin < 0:
            results[date_str] = {"ndvi": entry['ndvi'], "tmax": tmax, "tmin": tmin, "tavg": tavg, "Kc": None, "ETc": None}
            continue

        et0 = compute_et0_hargreaves(tmin, tmax, tavg, ra)
        stage_idx = get_stage_for_day(crop, sowing_date, date_obj)
        kc = get_kc_for_stage(crop, stage_idx)
        etc = et0 * kc

        results[date_str] = {"ndvi": entry['ndvi'], "tmax": tmax, "tmin": tmin, "tavg": tavg, "Kc": round(kc, 3), "ETc": round(etc, 4)}
    return results




@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        img_bytes = io.BytesIO(contents)

        image = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])

        preds = model.predict(input_arr)
        if preds.ndim == 2:
            probs = tf.nn.softmax(preds[0]).numpy()
        else:
            probs = tf.nn.softmax(preds).numpy().squeeze()

        pred_index = int(np.argmax(probs))
        pred_prob = float(probs[pred_index])
        class_lbl = class_name[pred_index] if 0 <= pred_index < len(class_name) else "unknown"

        return {
            "class_name": class_lbl,
            "pred_index": pred_index,
            "probability": round(pred_prob, 6)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    
@app.post("/predict_disease/insight")
def get_disease_insight(disease_info: dict = Body(...)):
    disease_name = disease_info.get("disease_name")
    if not disease_name:
        raise HTTPException(status_code=400, detail="disease_name is required")

    try:
        prompt = (
            f"You are an agricultural expert. Explain about the plant disease '{disease_name}': "
            "Provide a simple explanation of what this disease is, how it occurs and spreads, "
            "and how it can be treated, including recommended chemicals or fungicides. "
            "Keep the answer concise, clear, and practical."
        )

        model_genai = genai.GenerativeModel("gemini-2.5-flash")
        response = model_genai.generate_content(prompt)
        insight_text = (response.text or "").strip()
        if not insight_text:
            insight_text = "No insights generated by Gemini."

        return {"insight": insight_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")
    
class NDVISample(BaseModel):
    Date: str
    NDVI: float
    Tmax: Optional[float] = None
    Tmin: Optional[float] = None
    Tavg: Optional[float] = None
    Kc: Optional[float] = None
    ETc: Optional[float] = None


class NDVIInsightsRequest(BaseModel):
    crop: str
    ndvi_samples: List[NDVISample]

def generate_ndvi_etc_graph(ndvi_samples: List[NDVISample]):
    dates = [sample.Date for sample in ndvi_samples]
    ndvi_values = [sample.NDVI for sample in ndvi_samples]
    etc_values = [sample.ETc for sample in ndvi_samples]

    fig, ax1 = plt.subplots(figsize=(10,5))

    color = 'tab:green'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('NDVI', color=color)
    ax1.plot(dates, ndvi_values, color=color, marker='o', label='NDVI')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('ETc (mm)', color=color)
    ax2.plot(dates, etc_values, color=color, marker='x', label='ETc')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# ---------------- Endpoint ----------------
@app.post("/ndvi_insights_graph")
def ndvi_insights_graph(req: NDVIInsightsRequest):
    try:
        if req.crop not in GLOBAL_NDVI_ETC_STORAGE:
            raise HTTPException(status_code=400, detail=f"No NDVI/ETc data found for crop {req.crop}")

        stored_data = GLOBAL_NDVI_ETC_STORAGE[req.crop]
        ndvi_samples_list = stored_data["ndvi_samples"]

        # Convert to NDVISample objects if needed
        ndvi_samples = [
            NDVISample(
                Date=item['Date'],
                NDVI=item['NDVI'],
                Tmax=item.get('Tmax'),
                Tmin=item.get('Tmin'),
                Tavg=item.get('Tavg'),
                Kc=item.get('Kc'),
                ETc=item.get('ETc')
            ) for item in ndvi_samples_list
        ]
        # ------------------------------------------------------------------------------

        # Convert NDVISample objects into readable text format
        rows = []
        for sample in ndvi_samples:
            # safely extract numeric values with defaults
            ndvi_val = sample.NDVI if sample.NDVI is not None else 0
            etc_val  = sample.ETc if sample.ETc is not None else 0
            kc_val   = sample.Kc if sample.Kc is not None else 0
            tmax_val = sample.Tmax if sample.Tmax is not None else 0
            tmin_val = sample.Tmin if sample.Tmin is not None else 0
            tavg_val = sample.Tavg if sample.Tavg is not None else 0

            # create one line per date
            rows.append(
                f"{sample.Date}: NDVI={ndvi_val:.3f}, Tmax={tmax_val:.1f}, Tmin={tmin_val:.1f}, "
                f"Tavg={tavg_val:.1f}, Kc={kc_val:.2f}, ETc={etc_val:.2f} mm"
            )

        # Join all rows into one text block
        data_text = "\n".join(rows)

        # ---------------- Gemini prompt ----------------
        prompt = f"""
You are an experienced agronomist and remote-sensing specialist analyzing NDVI and ETc data for the crop {req.crop}.
Use the dataset below (each row represents a field observation date) to generate insights that can support practical field decisions.

Your analysis should help in:
- Understanding overall crop growth stage and vigor
- Identifying any stress periods or abnormal NDVI fluctuations
- Linking ETc variations with possible irrigation or climatic effects
- Recommending water or crop management actions
- Planning future field operations for the ongoing season

**Guidelines:**
- Focus entirely on interpreting the data patterns and practical implications.
- Do NOT explain what NDVI, ETc, Tmax, Tmin, Tavg, or Kc mean.
- Assume all ETc values are valid unless clearly missing.
- Write in clear, natural language — as if you personally generated and studied this data for the user's own field.
- Highlight any anomalies, peaks, or sudden declines that may indicate stress, over-irrigation, or recovery.
- Provide concise, professional recommendations in short bullet points or paragraphs.
- Do not add disclaimers about missing data unless ETc or NDVI are truly absent.

Format your answer as natural language (plain text), NOT JSON.

Here is the data:
{data_text}
"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        insights_text = (response.text or "").strip()
        if not insights_text:
            insights_text = "No insights generated by Gemini."

        # generate graph
        graph_base64 = generate_ndvi_etc_graph(req.ndvi_samples)

        return {
            "insights_text": insights_text,
            "graph_base64": graph_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini analysis or graph generation failed: {e}")

    

from fastapi.middleware.cors import CORSMiddleware    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)