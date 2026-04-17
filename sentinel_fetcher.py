"""
Sentinel-2 Imagery Fetching Agent
Uses the Copernicus Data Space Ecosystem (CDSE) STAC API for searching
satellite imagery. Falls back to synthetic demo tiles when credentials
are unavailable.
"""

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime, timedelta
import io
import os
import json


# Copernicus Data Space STAC Catalog (free, no auth for search)
STAC_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/stac"
STAC_SEARCH_URL = f"{STAC_CATALOG_URL}/search"

# Optional: set these env vars for actual image download
COPERNICUS_CLIENT_ID = os.environ.get("COPERNICUS_CLIENT_ID", "")
COPERNICUS_CLIENT_SECRET = os.environ.get("COPERNICUS_CLIENT_SECRET", "")


def search_sentinel2_scenes(bbox, date_str):
    """
    Search the Copernicus STAC catalog for Sentinel-2 L2A scenes using "Sandwich" logic.
    - Pre-Event: -30 days to 0 days, < 5% cloud cover.
    - Post-Event: 0 days to +15 days, < 10% cloud cover.
    - Validation Logic: Expands radius by +7 days if none found.
    """
    try:
        disaster_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            disaster_date = datetime.strptime(date_str, "%d-%m-%Y")
        except ValueError:
            raise ValueError(f"Invalid date string format: {date_str}. Must be YYYY-MM-DD or DD-MM-YYYY")
    
    pre_window = 30
    post_window = 15
    max_retries = 3
    
    results = {"pre_scenes": [], "post_scenes": []}
    
    for attempt in range(max_retries):
        # Calculate strict windows
        pre_start = (disaster_date - timedelta(days=pre_window)).strftime("%Y-%m-%dT00:00:00Z")
        pre_end = (disaster_date - timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")
        
        post_start = disaster_date.strftime("%Y-%m-%dT00:00:00Z")
        post_end = (disaster_date + timedelta(days=post_window)).strftime("%Y-%m-%dT23:59:59Z")
        
        searches = [
            ("pre_scenes", pre_start, pre_end, 5),     # < 5% clouds for pre
            ("post_scenes", post_start, post_end, 10)  # < 10% clouds for post
        ]
        
        found_both = True
        
        for period_key, start, end, max_cloud in searches:
            if results[period_key]: # already found good scene, keep it
                continue
                
            try:
                payload = {
                    "collections": ["sentinel-2-l2a"],
                    "bbox": bbox,
                    "datetime": f"{start}/{end}",
                    "limit": 5,
                    "query": {
                        "eo:cloud_cover": {"lte": max_cloud}
                    }
                }
                resp = requests.post(STAC_SEARCH_URL, json=payload, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    features = data.get("features", [])
                    if features:
                        # Sort by most recent to date and lowest cloud cover
                        features.sort(key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100))
                        results[period_key] = features
                    else:
                        found_both = False
                else:
                    found_both = False
            except Exception as e:
                print(f"[SentinelFetcher] STAC search failed for {period_key}: {e}")
                found_both = False
                
        if found_both:
            print(f"[SentinelFetcher] Found ideal cloudy-free scenes on attempt {attempt+1}")
            break
            
        print(f"[SentinelFetcher] Missing imagery. Expanding search window +7 days (Pre: {pre_window+7}, Post: {post_window+7})")
        pre_window += 7
        post_window += 7
        
    return results


def download_sentinel_tile(scene, bbox, size=1024):
    """
    Download an actual Sentinel-2 tile using CDSE access token.
    Requires COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET env vars.
    Returns a PIL Image or None if download fails.
    """
    if not COPERNICUS_CLIENT_ID or not COPERNICUS_CLIENT_SECRET:
        return None
    
    try:
        # Get access token
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        token_resp = requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": COPERNICUS_CLIENT_ID,
            "client_secret": COPERNICUS_CLIENT_SECRET
        }, timeout=10)
        
        if token_resp.status_code != 200:
            return None
            
        access_token = token_resp.json()["access_token"]
        
        # Use WMS/WMTS to get the tile rendered
        scene_id = scene.get("id", "")
        wms_url = "https://sh.dataspace.copernicus.eu/ogc/wms"
        
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "FORMAT": "image/png",
            "LAYERS": "TRUE-COLOR-S2L2A",
            "CRS": "EPSG:4326",
            "BBOX": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",
            "WIDTH": size,
            "HEIGHT": size,
            "TIME": scene["properties"]["datetime"][:10]
        }
        
        headers = {"Authorization": f"Bearer {access_token}"}
        img_resp = requests.get(wms_url, params=params, headers=headers, timeout=30)
        
        if img_resp.status_code == 200:
            return Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            
    except Exception as e:
        print(f"[SentinelFetcher] Download failed: {e}")
    
    return None


def generate_demo_tile(bbox, period="pre", disaster_type="earthquake", size=1024):
    """
    Generate a realistic-looking synthetic satellite tile for demo purposes.
    Uses procedural generation to simulate urban/vegetation areas.
    """
    np.random.seed(hash(f"{bbox}{period}") % (2**31))
    
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    if disaster_type in ["wildfire", "3"]:
        # Vegetation-heavy scene
        if period == "pre":
            # Lush green vegetation
            base_g = np.random.randint(80, 140, (size, size), dtype=np.uint8)
            base_r = (base_g * 0.4).astype(np.uint8)
            base_b = (base_g * 0.3).astype(np.uint8)
            img[:, :, 0] = base_r
            img[:, :, 1] = base_g
            img[:, :, 2] = base_b
            # Add some roads / clearings
            for _ in range(8):
                y = np.random.randint(0, size)
                thickness = np.random.randint(2, 6)
                img[max(0, y-thickness):min(size, y+thickness), :] = [160, 155, 140]
        else:
            # Post-wildfire: burn scars
            base = np.random.randint(40, 80, (size, size), dtype=np.uint8)
            img[:, :, 0] = (base * 1.5).clip(0, 255).astype(np.uint8)
            img[:, :, 1] = (base * 0.6).astype(np.uint8)
            img[:, :, 2] = (base * 0.3).astype(np.uint8)
            # Remaining green patches
            for _ in range(15):
                cx, cy = np.random.randint(100, size-100, 2)
                r = np.random.randint(20, 80)
                yy, xx = np.ogrid[-cy:size-cy, -cx:size-cx]
                mask = xx**2 + yy**2 <= r**2
                img[mask, 1] = np.random.randint(90, 130)
                img[mask, 0] = np.random.randint(30, 60)
    else:
        # Urban scene (earthquake, flood, hurricane, tsunami)
        if period == "pre":
            # Urban area with buildings
            img[:, :] = [180, 175, 165]  # Light urban base
            # Grid of buildings
            block_size = 40
            for bx in range(10, size - 10, block_size + 10):
                for by in range(10, size - 10, block_size + 8):
                    w = np.random.randint(20, block_size)
                    h = np.random.randint(15, block_size - 5)
                    shade = np.random.randint(100, 200)
                    img[by:by+h, bx:bx+w] = [shade, shade-10, shade-20]
            # Roads
            for i in range(0, size, block_size + 10):
                img[:, max(0, i-2):i+2] = [90, 90, 95]
                img[max(0, i-2):i+2, :] = [90, 90, 95]
        else:
            # Post-disaster: rubble, collapsed
            img[:, :] = [150, 140, 130]
            block_size = 40
            for bx in range(10, size - 10, block_size + 10):
                for by in range(10, size - 10, block_size + 8):
                    w = np.random.randint(20, block_size)
                    h = np.random.randint(15, block_size - 5)
                    if np.random.random() > 0.4:
                        shade = np.random.randint(60, 120)
                        img[by:by+h, bx:bx+w] = [shade+30, shade, shade-10]
                    else:
                        shade = np.random.randint(130, 200)
                        img[by:by+h, bx:bx+w] = [shade, shade-10, shade-20]
    
    # Add noise for realism
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return pil_img


def fetch_imagery(bbox, date_str, disaster_type="earthquake"):
    """
    Main entry point: Fetch pre and post disaster satellite imagery.
    Tries real Sentinel-2 API first, falls back to demo tiles.
    
    Args:
        bbox: [west, south, east, north]
        date_str: 'YYYY-MM-DD'
        disaster_type: one of 'earthquake', 'flood', 'hurricane', 'wildfire', 'tsunami'
    
    Returns:
        dict with 'pre_image' (PIL), 'post_image' (PIL), 'metadata' (dict)
    """
    metadata = {
        "bbox": bbox,
        "disaster_date": date_str,
        "disaster_type": disaster_type,
        "source": "demo",
        "pre_date": "",
        "post_date": ""
    }
    
    pre_image = None
    post_image = None
    
    # Step 1: Try real Sentinel-2 API
    print(f"[SentinelFetcher] Searching for imagery around {date_str} in bbox {bbox}...")
    scenes = search_sentinel2_scenes(bbox, date_str)
    
    if scenes["pre_scenes"]:
        pre_image = download_sentinel_tile(scenes["pre_scenes"][0], bbox)
        if pre_image:
            metadata["source"] = "sentinel-2"
            metadata["pre_date"] = scenes["pre_scenes"][0]["properties"]["datetime"][:10]
            metadata["pre_cloud_cover"] = scenes["pre_scenes"][0]["properties"].get("eo:cloud_cover", "N/A")
            
    if scenes["post_scenes"]:
        post_image = download_sentinel_tile(scenes["post_scenes"][0], bbox)
        if post_image:
            metadata["post_date"] = scenes["post_scenes"][0]["properties"]["datetime"][:10]
            metadata["post_cloud_cover"] = scenes["post_scenes"][0]["properties"].get("eo:cloud_cover", "N/A")
    
    # Step 2: Fall back to demo tiles
    if pre_image is None:
        try:
            disaster_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            disaster_date = datetime.strptime(date_str, "%d-%m-%Y")
        metadata["pre_date"] = (disaster_date - timedelta(days=12)).strftime("%Y-%m-%d")
        metadata["post_date"] = (disaster_date + timedelta(days=3)).strftime("%Y-%m-%d")
        metadata["source"] = "demo"
        pre_image = generate_demo_tile(bbox, "pre", disaster_type)
        
    if post_image is None:
        post_image = generate_demo_tile(bbox, "post", disaster_type)
    
    print(f"[SentinelFetcher] Source: {metadata['source']} | Pre: {metadata['pre_date']} | Post: {metadata['post_date']}")
    
    return {
        "pre_image": pre_image,
        "post_image": post_image,
        "metadata": metadata
    }
