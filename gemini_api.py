import os
import google.generativeai as genai
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime
import base64
import json

# Simple in-memory cache
_product_analysis_cache = {}  # Format: {image_path: {result_data}}

def _get_cache_key(product_image, mrp_image):
    """Generate a cache key based on the image paths."""
    return f"{product_image}:{mrp_image}"

def _store_in_cache(key, result):
    """Store result in the in-memory cache."""
    global _product_analysis_cache
    _product_analysis_cache[key] = result
    
    # Limit cache size to prevent memory issues
    if len(_product_analysis_cache) > 100:
        # Remove the oldest entry (first key)
        if _product_analysis_cache:
            oldest_key = next(iter(_product_analysis_cache))
            del _product_analysis_cache[oldest_key]

def _get_from_cache(key):
    """Get result from in-memory cache."""
    return _product_analysis_cache.get(key)

# Initialize Gemini API
def initialize_gemini(api_key):
    genai.configure(api_key=api_key)
    # Using gemini-2.0-flash instead of gemini-pro-vision
    return genai.GenerativeModel('gemini-2.0-flash')

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to extract product details using Gemini API
async def extract_product_details(product_image_path, mrp_image_path, model):
    try:
        # Check if files exist
        if not os.path.exists(product_image_path):
            raise FileNotFoundError(f"Product image not found: {product_image_path}")
        
        # Check cache first
        cache_key = _get_cache_key(product_image_path, mrp_image_path)
        cached_result = _get_from_cache(cache_key)
        
        if cached_result:
            print(f"Using cached product analysis for {cache_key}")
            return cached_result
            
        # Prepare prompt based on whether we have an MRP image or not
        # Check if mrp_image_path is not None and if the file exists
        has_mrp_image = mrp_image_path is not None and os.path.exists(mrp_image_path)
        
        if has_mrp_image:
            # For products with MRP tag
            prompt = """
            Analyze these product images and extract the following information:
            
            1. Type of product: Packed/Unpacked
            2. If unpacked, specify if it's a fruit/vegetable
            3. Name of product
            4. If unpacked, assess freshness: Fresh/Rotten
            5. If unpacked, estimate shelf life in days
            6. If packed, identify the MRP (price) from the MRP tag image
            7. If packed, identify the manufacturing date from the MRP tag image
            8. If packed, identify the expiry date from the MRP tag image
            
            The first image is the product, and the second image is the close-up of the MRP tag.
            
            Format your response in JSON with these exact keys:
            {
              "product_type": "packed/unpacked",
              "unpacked_type": "fruit/vegetable/N/A",
              "product_name": "name of product",
              "freshness": "fresh/rotten/N/A",
              "shelf_life_days": number or "N/A",
              "mrp": "price or N/A",
              "manufacturing_date": "date or N/A",
              "expiry_date": "date or N/A"
            }
            """
        else:
            # For products without MRP tag (likely fruits/vegetables)
            prompt = """
            Analyze this product image and extract the following information:
            
            1. Type of product: Packed/Unpacked (this appears to be an unpacked product)
            2. Specify if it's a fruit/vegetable and what type
            3. Name of product
            4. Assess freshness: Fresh/Rotten
            5. Estimate shelf life in days for this product
            
            Format your response in JSON with these exact keys:
            {
              "product_type": "unpacked",
              "unpacked_type": "fruit/vegetable",
              "product_name": "name of product",
              "freshness": "fresh/rotten",
              "shelf_life_days": number,
              "mrp": "N/A",
              "manufacturing_date": "N/A",
              "expiry_date": "N/A"
            }
            """

        # Create input for the model
        contents = [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": encode_image(product_image_path)
                    }}
                ]
            }
        ]
        
        # Add MRP image if available and valid
        if has_mrp_image:
            contents[0]["parts"].append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": encode_image(mrp_image_path)
                }
            })
        
        print(f"Calling Gemini API for {product_image_path}")
        # Generate response from Gemini
        response = model.generate_content(contents)
        
        # Extract JSON from response
        try:
            # This assumes the model returns valid JSON. If not, we'll need to parse it.
            response_text = response.text
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                # If no JSON found, return error
                raise ValueError("No valid JSON found in model response")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from model response: {response.text}")
        
        # Add timestamp and source image paths to result
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["product_image"] = product_image_path
        result["mrp_image"] = mrp_image_path
        
        # Store in cache
        _store_in_cache(cache_key, result)
        
        return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "product_image": product_image_path,
            "mrp_image": mrp_image_path if 'mrp_image_path' in locals() else None
        }
    

def store_product_info(product_info):
    """Store the extracted product information."""
    # Create directory if it doesn't exist
    os.makedirs("static/product_info", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"static/product_info/product_{timestamp}.json"
    
    # Store in cache
    cache_key = _get_cache_key(product_info.get("product_image"), product_info.get("mrp_image"))
    _store_in_cache(cache_key, product_info)
    
    # Write to file
    with open(filename, "w") as f:
        json.dump(product_info, f, indent=2)
    # Log success
    print(f"Product information saved to {filename}")
    
    return filename

# Function to add to FastAPI app
def add_gemini_routes(app, api_key):
    model = initialize_gemini(api_key)
    
    @app.get("/analyze_product")
    async def analyze_product():
        """Analyze the latest captured products from all cameras."""
        from app import best_detection_result, last_captured_images, debug_log
        
        debug_log("Starting analyze_product API call")
        
        # First, try to use the best detection if available
        if best_detection_result:
            # Get best MRP detection image
            mrp_image_path = best_detection_result.get("roi_image")
            
            # Get product image from the same camera
            camera_id = best_detection_result.get("camera_id")
            product_image_path = last_captured_images.get(camera_id)
            
            debug_log(f"Found MRP detection, using camera {camera_id} for analysis")
        else:
            # If no MRP detection available, just use the first camera's image
            if last_captured_images:
                camera_id = next(iter(last_captured_images))
                product_image_path = last_captured_images.get(camera_id)
                mrp_image_path = None
                debug_log(f"No MRP detection found, using camera {camera_id} image for analysis")
            else:
                debug_log("No captured images available for analysis")
                return {"message": "No captured images available", "status": "warning"}
        
        if not product_image_path:
            debug_log("No product image available")
            return {"message": "No product image available", "status": "warning"}
        
        # Verify product image file exists
        if not os.path.exists(product_image_path):
            debug_log(f"Product image file not found: {product_image_path}")
            return {"error": f"Product image file not found: {product_image_path}", "status": "error"}
        
        # Check if MRP image exists, set to None if it doesn't
        if mrp_image_path and not os.path.exists(mrp_image_path):
            debug_log(f"MRP image file not found: {mrp_image_path}, continuing with just product image")
            mrp_image_path = None
        
        # Check cache first
        cache_key = _get_cache_key(product_image_path, mrp_image_path)
        cached_result = _get_from_cache(cache_key)
        
        if cached_result:
            debug_log(f"Using cached product analysis for {cache_key}")
            return cached_result
        
        # Check if a cached result exists for this image on disk
        cache_dir = "static/product_info"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if a cached result exists for this image
        cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
        for file in cached_files:
            try:
                with open(os.path.join(cache_dir, file), 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get("product_image") == product_image_path and cached_data.get("mrp_image") == mrp_image_path:
                        debug_log(f"Using disk-cached product analysis for {cache_key}")
                        # Also add to memory cache
                        _store_in_cache(cache_key, cached_data)
                        return cached_data
            except Exception as e:
                debug_log(f"Error reading cache file {file}: {str(e)}")
                continue
        
        # If not in cache, extract product details
        debug_log(f"Performing new product analysis for {cache_key}")
        result = await extract_product_details(product_image_path, mrp_image_path, model)
        
        # Store result in cache
        _store_in_cache(cache_key, result)
        
        # Store to disk
        store_product_info(result)
        
        debug_log("Product analysis complete and cached")
        
        return result
    
    # This endpoint will be called when ESP32 triggers a capture
    @app.post("/process_capture")
    async def process_capture():
        """Process the latest capture and extract product details."""
        from app import capture_all, debug_log
        
        debug_log("Starting process_capture endpoint")
        
        # The capture_all function now handles waiting for YOLO and running Gemini analysis
        # It will perform product analysis only once and return the result
        capture_result = await capture_all()
        
        # Return the results directly
        if "product_analysis" in capture_result:
            debug_log("Product analysis completed successfully in process_capture")
            return capture_result["product_analysis"]
        elif "product_analysis_error" in capture_result:
            debug_log(f"Error in product analysis: {capture_result['product_analysis_error']}")
            return {
                "error": capture_result["product_analysis_error"],
                "message": "Error occurred during product analysis", 
                "status": "error"
            }
        else:
            debug_log("No product analysis was performed")
            return {
                "message": "No product analysis was performed. Check camera and detection status.",
                "status": "warning"
            }