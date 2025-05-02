from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import threading
import time
import os
import uvicorn
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import json
import queue
import asyncio
import gc
from dotenv import load_dotenv

# Debug logging helper
def debug_log(message):
    """Log a debug message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[DEBUG {timestamp}] {message}")

# Load environment variables
load_dotenv()

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/captures", exist_ok=True)
os.makedirs("static/detections", exist_ok=True)
os.makedirs("static/product_info", exist_ok=True)

app = FastAPI(title="Computer Vision Based Inventory Management System")

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Global variables
camera_streams = {}  # Now stores URLs instead of OpenCV objects
camera_locks = {}
last_captured_images = {}
last_detection_results = {}
best_detection_result = None  # Will store the detection with highest confidence
yolo_model = None
processing_status = {}  # Track processing status
comparison_lock = threading.Lock()  # Lock for thread-safe comparison of detections

# For keeping the latest frame from each camera
latest_frames = {}
frame_timestamps = {}  # Store timestamps for each frame
frame_locks = {}  # Separate locks for frame access

# Create a processing queue for GPU tasks
gpu_task_queue = queue.Queue()
gpu_worker_running = False

# Flag to control camera reader threads
camera_threads_running = True

# Add a client counter to track active connections
active_stream_clients = {}
client_lock = threading.Lock()

# Configuration limits for memory management
MAX_STORED_DETECTIONS = 5  # Maximum number of detection results to keep
MAX_INACTIVE_TIME = 300  # Maximum time (seconds) to keep a connection without activity

class CameraConfig(BaseModel):
    camera1_url: str
    camera2_url: str

class YoloModelConfig(BaseModel):
    model_path: str = "models/train.pt"  # Default path to the YOLO model
    confidence: float = 0.1  # Lower default confidence threshold
    use_cpu: bool = False  # Option to use CPU instead of GPU

def clear_detection_for_camera(camera_id):
    """Clear the detection results for a specific camera and update best detection if needed."""
    global last_detection_results, best_detection_result, comparison_lock
    
    # Use lock for thread safety
    with comparison_lock:
        # Check if there was a previous detection for this camera
        if camera_id in last_detection_results:
            # Remove the detection for this camera
            last_detection_result = last_detection_results.pop(camera_id)
            
            # Check if we need to update the best detection
            if best_detection_result and best_detection_result.get("camera_id") == camera_id:
                print(f"Clearing best detection since camera {camera_id} no longer has a valid detection")
                
                # Find new best detection from remaining cameras
                new_best = None
                best_confidence = 0
                
                for cam_id, result in last_detection_results.items():
                    if result["confidence"] > best_confidence:
                        new_best = result
                        best_confidence = result["confidence"]
                
                if new_best:
                    # Update with new best detection
                    new_best["camera_id"] = cam_id
                    new_best["info"] = f"Detected by Camera {cam_id[-1]} with {new_best['confidence']*100:.2f}% confidence"
                    best_detection_result = new_best
                    print(f"New best detection set from {cam_id} with confidence {best_confidence}")
                else:
                    # No valid detections left
                    best_detection_result = None
                    print("No valid detections remaining, best detection cleared")

def load_yolo_model(model_path, use_cpu=False):
    """Load YOLO model from the specified path using Ultralytics."""
    try:
        # Import ultralytics here to avoid loading the model unnecessarily
        from ultralytics import YOLO
        
        print(f"Loading YOLO model from {model_path}")
        
        # Force garbage collection before loading model
        gc.collect()
        
        # Load the model
        if use_cpu:
            # Force CPU usage
            import torch
            with torch.device('cpu'):
                model = YOLO(model_path)
                # Ensure the model stays on CPU
                model.to('cpu')
            print("Using CPU for YOLO model")
        else:
            # Use default device (GPU if available)
            model = YOLO(model_path)
            print("Using default device for YOLO model")
        
        print(f"YOLO model loaded successfully from {model_path}")
        print(f"Model type: {type(model)}")
        print(f"Model task: {model.task}")
        print(f"Model classes: {model.names}")
        
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Camera reader thread function
def camera_reader(camera_id, url, buffer_size=1):
    """Continuously read frames from the camera in a separate thread."""
    global latest_frames, frame_timestamps, frame_locks, camera_threads_running, active_stream_clients
    
    # Create a dedicated VideoCapture object for this thread
    cap = cv2.VideoCapture(url)
    
    # Configure capture parameters for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)  # Set minimal buffer size
    
    # RTSP specific configuration (if applicable)
    if url.lower().startswith("rtsp"):
        # Use TCP for RTSP (more reliable than UDP)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        # You can adjust other RTSP specific parameters here if needed
    
    # Initialize latest frame storage
    with frame_locks[camera_id]:
        latest_frames[camera_id] = None
        frame_timestamps[camera_id] = 0
    
    frame_count = 0
    last_log_time = time.time()
    last_client_check = time.time()
    
    while camera_threads_running:
        try:
            # Check if we have any active clients - only read frames if needed
            current_time = time.time()
            
            # Periodically check if there are any active clients (every 5 seconds)
            if current_time - last_client_check >= 5:
                with client_lock:
                    has_clients = camera_id in active_stream_clients and active_stream_clients[camera_id] > 0
                last_client_check = current_time
                
                # If no clients are connected, sleep to reduce resource usage
                if not has_clients:
                    time.sleep(0.5)
                    continue
            
            # Read frame without holding the lock during potentially slow I/O operation
            success, frame = cap.read()
            
            if success:
                # Only update the frame if it's needed (has active clients or recent capture)
                with client_lock:
                    has_clients = camera_id in active_stream_clients and active_stream_clients[camera_id] > 0
                
                # Update the latest frame with a lock
                with frame_locks[camera_id]:
                    # If the frame is different enough from the previous one or we have active clients
                    if has_clients or latest_frames[camera_id] is None:
                        latest_frames[camera_id] = frame.copy()  # Store a copy to avoid race conditions
                        frame_timestamps[camera_id] = time.time()  # Update timestamp
                
                frame_count += 1
                
                # Log FPS every 5 seconds
                current_time = time.time()
                if current_time - last_log_time >= 5:
                    fps = frame_count / (current_time - last_log_time)
                    print(f"Camera {camera_id} FPS: {fps:.2f}, Active clients: {active_stream_clients.get(camera_id, 0)}")
                    frame_count = 0
                    last_log_time = current_time
                
                # Sleep a tiny amount to reduce CPU usage
                time.sleep(0.01)  # Slightly increased to reduce CPU usage
            else:
                # If read fails, log and retry after a short delay
                print(f"Camera {camera_id} read failed, reconnecting...")
                # Close and reopen the connection
                cap.release()
                time.sleep(1)  # Wait before reconnecting
                cap = cv2.VideoCapture(url)  # Reconnect
                
                # Reset counters
                frame_count = 0
                last_log_time = time.time()
        except Exception as e:
            print(f"Error in camera_reader for {camera_id}: {str(e)}")
            time.sleep(0.5)  # Wait before retrying
    
    # Clean up
    cap.release()
    print(f"Camera reader thread for {camera_id} stopped")

# Cleanup old files function
def cleanup_old_files(directory, max_age_hours=24, max_files=100):
    """Remove old files to prevent disk space issues."""
    try:
        if not os.path.exists(directory):
            return
            
        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        # Sort files by modification time (oldest first)
        files.sort(key=lambda x: os.path.getmtime(x))
        
        # Current time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Remove old files exceeding the maximum count
        if len(files) > max_files:
            for file_path in files[:-max_files]:
                try:
                    os.remove(file_path)
                    print(f"Removed old file: {file_path}")
                except Exception as e:
                    print(f"Error removing file {file_path}: {str(e)}")
        
        # Remove files older than the maximum age
        for file_path in files:
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f"Removed aged file: {file_path}")
                except Exception as e:
                    print(f"Error removing file {file_path}: {str(e)}")
    except Exception as e:
        print(f"Error in cleanup_old_files: {str(e)}")

# Start a background thread for file cleanup
def start_cleanup_thread():
    """Start a background thread to periodically clean up old files."""
    def cleanup_worker():
        while True:
            try:
                cleanup_old_files("static/captures")
                cleanup_old_files("static/detections")
                cleanup_old_files("static/product_info")
            except Exception as e:
                print(f"Error in cleanup worker: {str(e)}")
            # Run every hour
            time.sleep(3600)
    
    thread = threading.Thread(target=cleanup_worker, daemon=True)
    thread.start()

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Display the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/setup-cameras")
async def setup_cameras(config: CameraConfig):
    """Setup camera streams with the provided URLs."""
    global camera_streams, camera_locks, last_captured_images, best_detection_result
    global latest_frames, frame_timestamps, frame_locks, camera_threads_running, active_stream_clients
    
    # Stop existing camera threads if running
    camera_threads_running = False
    time.sleep(1)  # Allow threads to terminate
    
    # Clear dictionaries
    camera_streams.clear()
    camera_locks.clear()
    latest_frames.clear()
    frame_timestamps.clear()
    frame_locks.clear()
    active_stream_clients.clear()
    last_captured_images.clear()
    best_detection_result = None
    
    # Force garbage collection
    gc.collect()
    
    # Reset the flag for new threads
    camera_threads_running = True
    
    # Setup new streams
    try:
        urls = {
            "camera1": config.camera1_url,
            "camera2": config.camera2_url
        }
        
        # Initialize locks for each camera
        for camera_id in urls.keys():
            camera_locks[camera_id] = threading.Lock()
            frame_locks[camera_id] = threading.Lock()
            active_stream_clients[camera_id] = 0
        
        # Start camera reader threads
        for camera_id, url in urls.items():
            # Store the URL for reference
            camera_streams[camera_id] = url
            
            # Start a dedicated reader thread for each camera
            threading.Thread(
                target=camera_reader,
                args=(camera_id, url),
                daemon=True
            ).start()
        
        return {"message": "Cameras setup successfully", "status": "success"}
    except Exception as e:
        return {"message": f"Error setting up cameras: {str(e)}", "status": "error"}

@app.post("/setup-yolo-model")
async def setup_yolo_model(config: YoloModelConfig):
    """Load the YOLO model."""
    global yolo_model
    
    try:
        # Clear previous model to free up memory
        if yolo_model is not None:
            del yolo_model
            gc.collect()
        
        yolo_model = load_yolo_model(config.model_path, config.use_cpu)
        if yolo_model:
            return {"message": "YOLO model loaded successfully", "status": "success"}
        else:
            return {"message": "Failed to load YOLO model", "status": "error"}
    except Exception as e:
        return {"message": f"Error loading YOLO model: {str(e)}", "status": "error"}

def generate_frames(camera_id):
    """Generate frames from the camera's latest frame buffer."""
    global latest_frames, frame_timestamps, frame_locks, active_stream_clients
    
    # Register this client
    client_id = f"{camera_id}_{int(time.time() * 1000)}"
    
    try:
        # Add client to count
        with client_lock:
            if camera_id not in active_stream_clients:
                active_stream_clients[camera_id] = 0
            active_stream_clients[camera_id] += 1
            print(f"Client {client_id} connected to camera {camera_id}. Total clients: {active_stream_clients[camera_id]}")
        
        if camera_id not in frame_locks:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   b'No camera connected' + 
                   b'\r\n')
            return
        
        last_sent_timestamp = 0
        last_activity_time = time.time()
        
        while True:
            # Use a try-except block to handle any errors
            try:
                # Check if client has been inactive for too long
                current_time = time.time()
                if current_time - last_activity_time > MAX_INACTIVE_TIME:
                    print(f"Client {client_id} timed out due to inactivity")
                    break
                
                # Get the latest frame with a lock
                with frame_locks[camera_id]:
                    frame = latest_frames.get(camera_id)
                    current_timestamp = frame_timestamps.get(camera_id, 0)
                
                # Only send new frames to reduce bandwidth
                if frame is not None and current_timestamp > last_sent_timestamp:
                    # Encode the frame to JPEG format
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue
                    
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    last_sent_timestamp = current_timestamp
                    last_activity_time = current_time  # Update activity time when frame is sent
                
                # Short sleep to prevent high CPU usage
                time.sleep(0.03)  # Adjust as needed for desired frame rate
            except Exception as e:
                # If any error occurs, log and continue
                print(f"Error in generate_frames for {camera_id}: {str(e)}")
                time.sleep(0.1)
                
                # If the error persists, reset the counter
                last_sent_timestamp = 0
    finally:
        # Remove client from count
        with client_lock:
            if camera_id in active_stream_clients:
                active_stream_clients[camera_id] = max(0, active_stream_clients[camera_id] - 1)
                print(f"Client {client_id} disconnected from camera {camera_id}. Remaining clients: {active_stream_clients[camera_id]}")

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    """Stream video feed from the specified camera."""
    if camera_id not in camera_streams:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/capture/{camera_id}")
async def capture(camera_id: str):
    """Capture a photo from the specified camera."""
    global latest_frames, frame_locks, last_captured_images, processing_status
    
    if camera_id not in camera_streams:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Get the latest frame from the buffer
    frame = None
    with frame_locks[camera_id]:
        if latest_frames[camera_id] is not None:
            frame = latest_frames[camera_id].copy()
    
    if frame is None:
        return {"status": "error", "message": "No frame available for capture"}
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"static/captures/camera_{camera_id}_{timestamp}.jpg"
    
    # Save the image to disk
    cv2.imwrite(filename, frame)
    
    # Limit the number of stored images per camera
    # First, clear previous capture path to save memory
    if camera_id in last_captured_images:
        old_filename = last_captured_images[camera_id]
        # Don't delete the file as it might be referenced elsewhere
    
    # Store the image path
    last_captured_images[camera_id] = filename
    
    # Set processing status to in-progress
    processing_status[camera_id] = "processing"
    
    # Process the image with YOLO model if available
    if yolo_model:
        # Add task to queue instead of starting a new thread directly
        add_to_gpu_queue(filename, camera_id)
    else:
        print("YOLO model not available. Please load the model first.")
        processing_status[camera_id] = "no_model"
    
    return {
        "status": "success", 
        "message": "Photo captured successfully", 
        "filename": filename,
        "detection_available": camera_id in last_detection_results
    }

def add_to_gpu_queue(image_path, camera_id):
    """Add a task to the GPU processing queue and start the worker if not running."""
    global gpu_worker_running
    
    # Add the task to the queue
    gpu_task_queue.put((image_path, camera_id))
    
    # Start the worker thread if not already running
    if not gpu_worker_running:
        threading.Thread(target=gpu_worker, daemon=True).start()

def gpu_worker():
    """Worker thread that processes GPU tasks one at a time."""
    global gpu_worker_running
    
    gpu_worker_running = True
    
    try:
        while True:
            # Get a task from the queue, with a timeout
            try:
                image_path, camera_id = gpu_task_queue.get(timeout=5)
                print(f"Processing image from queue: {image_path} for camera {camera_id}")
                
                # Process the image
                process_image_with_yolo(image_path, camera_id)
                
                # Explicitly run garbage collection after processing
                gc.collect()
                
                # Mark the task as done
                gpu_task_queue.task_done()
                
            except queue.Empty:
                # Queue is empty, wait a bit before checking again
                print("GPU queue is empty, worker waiting...")
                time.sleep(1)
                
                # If queue is still empty after waiting, exit worker
                if gpu_task_queue.empty():
                    print("GPU worker shutting down due to inactivity")
                    break
    finally:
        gpu_worker_running = False

def update_best_detection(camera_id, result_data):
    """Update the best detection result if current one has higher confidence."""
    global best_detection_result
    
    # Use a lock to ensure thread safety
    with comparison_lock:
        # Add camera_id to the detection data
        result_data["camera_id"] = camera_id
        
        # If this is the first detection or has higher confidence than current best
        if best_detection_result is None or result_data["confidence"] > best_detection_result["confidence"]:
            print(f"New best detection from {camera_id} with confidence {result_data['confidence']}")
            best_detection_result = result_data
            
            # Create combined string for additional info
            best_detection_result["info"] = f"Detected by Camera {camera_id[-1]} with {result_data['confidence']*100:.2f}% confidence"

def limit_detection_results(max_results=MAX_STORED_DETECTIONS):
    """Limit the number of stored detection results to prevent memory growth."""
    global last_detection_results
    
    with comparison_lock:
        # If we have more than max_results, remove the oldest ones
        if len(last_detection_results) > max_results:
            # Get the keys sorted by timestamp (oldest first)
            sorted_keys = sorted(
                last_detection_results.keys(),
                key=lambda k: last_detection_results[k].get("timestamp", "")
            )
            
            # Remove the oldest entries
            for key in sorted_keys[:(len(sorted_keys) - max_results)]:
                del last_detection_results[key]
                print(f"Removed old detection result for {key} due to limit")

def process_image_with_yolo(image_path, camera_id):
    """Process the captured image with YOLO model to detect MRP tags."""
    global yolo_model, last_detection_results, processing_status
    
    try:
        if yolo_model is None:
            print("Warning: YOLO model not loaded, skipping detection")
            processing_status[camera_id] = "no_model"
            return
        
        print(f"Starting YOLO processing for {image_path}")
        processing_status[camera_id] = "processing"
        
        # Set a lower confidence threshold for better detection chances
        conf_threshold = 0.1
        print(f"Using confidence threshold: {conf_threshold}")
        
        # Run inference with the Ultralytics YOLO model
        # Add error handling with retry for CUDA errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                results = yolo_model(image_path, conf=conf_threshold)
                break  # If successful, break the retry loop
            except RuntimeError as e:
                if "CUDA" in str(e) and attempt < max_retries - 1:
                    print(f"CUDA error on attempt {attempt+1}, retrying after delay...")
                    time.sleep(2)  # Wait before retry
                    gc.collect()  # Force garbage collection
                else:
                    # Re-raise the error if we've exhausted retries or it's not a CUDA error
                    raise
        
        # Check that results is not empty
        if results is None or len(results) == 0:
            print("Warning: YOLO model returned empty results")
            processing_status[camera_id] = "no_results"
            
            # Save a debug image
            debug_img = cv2.imread(image_path)
            debug_filename = f"static/detections/debug_empty_{camera_id}_{os.path.basename(image_path)}"
            cv2.imwrite(debug_filename, debug_img)
            print(f"Saved debug image to {debug_filename}")
            return
        
        print(f"YOLO returned {len(results)} result objects")
        
        # Process results
        result = results[0]  # First result object
        
        # Save the debug visualization regardless of detections
        debug_filename = f"static/detections/debug_{camera_id}_{os.path.basename(image_path)}"
        debug_img = result.plot()  # Get visualization from YOLO
        cv2.imwrite(debug_filename, debug_img)
        print(f"Saved debug visualization to {debug_filename}")
        
        # Check if any boxes were detected
        if len(result.boxes) > 0:
            print(f"Found {len(result.boxes)} detections")
            
            # Load the original image for visualization
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image at {image_path}")
                processing_status[camera_id] = "error"
                return
            
            # Convert boxes to a list of dictionaries for easier handling
            detections = []
            for i, box in enumerate(result.boxes):
                # Extract box information
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    print(f"Detection {i}: Class={class_name}, Conf={confidence:.4f}, Box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    
                    detections.append({
                        'xmin': int(x1),
                        'ymin': int(y1),
                        'xmax': int(x2),
                        'ymax': int(y2),
                        'confidence': confidence,
                        'name': class_name
                    })
                except Exception as e:
                    print(f"Error processing detection {i}: {e}")
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Get the detection with highest confidence
            best_detection = detections[0]
            
            # Create a copy of the image to draw on
            img_with_box = img.copy()
            
            # Draw bounding box on the image
            x1, y1, x2, y2 = best_detection['xmin'], best_detection['ymin'], best_detection['xmax'], best_detection['ymax']
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"{best_detection['name']}: {best_detection['confidence']:.2f}"
            cv2.putText(img_with_box, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the image with detection
            detection_filename = f"static/detections/detection_{camera_id}_{os.path.basename(image_path)}"
            cv2.imwrite(detection_filename, img_with_box)
            
            # Crop the region of interest (ROI)
            roi = img[y1:y2, x1:x2]
            roi_filename = f"static/detections/roi_{camera_id}_{os.path.basename(image_path)}"
            cv2.imwrite(roi_filename, roi)
            
            # Store detection results
            result_data = {
                "original_image": image_path,
                "detection_image": detection_filename,
                "roi_image": roi_filename,
                "confidence": float(best_detection['confidence']),
                "class": best_detection['name'],
                "bbox": [x1, y1, x2, y2],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Limit detection results count before adding new one
            limit_detection_results()
            
            # Store the result for this camera
            last_detection_results[camera_id] = result_data
            
            # Update the best detection across all cameras
            update_best_detection(camera_id, result_data)
            
            processing_status[camera_id] = "complete"
            
            print(f"Successfully processed image with YOLO. Detection saved to {detection_filename}")
        else:
            print("No detections found in the image")
            processing_status[camera_id] = "no_detections"
            
            # Save the debug image with a different name
            debug_img = cv2.imread(image_path)
            debug_filename = f"static/detections/no_detection_{camera_id}_{os.path.basename(image_path)}"
            cv2.imwrite(debug_filename, debug_img)
            print(f"Saved no-detection image to {debug_filename}")
            
            # Clear previous detection for this camera since no new detection was found
            clear_detection_for_camera(camera_id)
            
    except Exception as e:
        print(f"Error processing image with YOLO: {str(e)}")
        import traceback
        traceback.print_exc()
        processing_status[camera_id] = "error"
    finally:
        # Explicitly delete local variables that might hold large data
        if 'results' in locals():
            del results
        if 'img' in locals():
            del img
        if 'img_with_box' in locals():
            del img_with_box
        if 'debug_img' in locals():
            del debug_img
        if 'roi' in locals():
            del roi
        
        # Force garbage collection
        gc.collect()

@app.get("/capture")
async def capture_all():
    """Capture photos from all cameras - This endpoint will be called by ESP32."""
    global best_detection_result, GEMINI_API_KEY
    
    debug_log("Starting capture_all API call")
    results = {}
    product_analyzed = False  # Add a flag to track if product analysis has been done
    
    # Process cameras one at a time to avoid CUDA conflicts
    for camera_id in camera_streams.keys():
        debug_log(f"Capturing from camera {camera_id}")
        # Call the individual capture endpoint for each camera
        result = await capture(camera_id)
        results[camera_id] = result
        
        # Small delay between captures to avoid CUDA conflicts
        time.sleep(0.5)
    
    # Wait for YOLO processing to complete before attempting Gemini analysis
    if GEMINI_API_KEY and not product_analyzed:
        debug_log("Waiting for YOLO processing to complete before Gemini analysis")
        # Wait for detection to complete
        await asyncio.sleep(2)  # Give YOLO processing time to start
        
        # Wait for processing to complete (simple polling implementation)
        max_wait_time = 15  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            statuses = list(processing_status.values())
            all_complete = all(status == "complete" or status == "no_detections" or status == "error" for status in statuses)
            
            if all_complete:
                debug_log("YOLO processing complete, proceeding with Gemini analysis")
                break
                
            debug_log(f"Waiting for YOLO processing: {statuses}")
            await asyncio.sleep(0.5)
            
            # If we've waited more than 5 seconds, run garbage collection
            if time.time() - start_time > 5 and (time.time() - start_time) % 5 < 0.5:
                gc.collect()
        
        # Now attempt to analyze the product - ONLY ONCE
        from gemini_api import extract_product_details
        
        # Select the best image for analysis
        # First try to use an image with MRP detection if available
        if best_detection_result:
            # Get images for analysis (from the camera with the best detection)
            camera_id = best_detection_result.get("camera_id")
            product_image = last_captured_images.get(camera_id)
            mrp_image = best_detection_result.get("roi_image")
            debug_log(f"Using best detection from camera {camera_id} for analysis")
        else:
            # If no MRP detections, just use the first camera's image
            # You can implement more sophisticated camera selection logic here if needed
            if last_captured_images:
                camera_id = next(iter(last_captured_images))
                product_image = last_captured_images.get(camera_id)
                mrp_image = None
                debug_log(f"No MRP detection found, using camera {camera_id} image for analysis")
            else:
                debug_log("No captured images available for analysis")
                product_image = None
                mrp_image = None
        
        # Check if MRP image exists, set to None if it doesn't
        if mrp_image and not os.path.exists(mrp_image):
            debug_log(f"MRP image file not found: {mrp_image}, continuing with just product image")
            mrp_image = None
            
        # Import the model from gemini_api
        if product_image and os.path.exists(product_image):
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Extract product details
            try:
                debug_log(f"Starting Gemini analysis for product: {product_image}")
                analysis_result = await extract_product_details(product_image, mrp_image, model)
                
                # Store the result
                results["product_analysis"] = analysis_result
                
                # Store results in a file
                from gemini_api import store_product_info
                store_product_info(analysis_result)
                
                debug_log("Product analysis complete and saved")
                product_analyzed = True  # Set flag to indicate analysis is done
            except Exception as e:
                debug_log(f"Error in Gemini product analysis: {str(e)}")
                results["product_analysis_error"] = str(e)
            finally:
                # Clean up memory
                del model
                gc.collect()
        else:
            if not product_image:
                debug_log("No product image available for Gemini analysis")
                results["analysis_status"] = "skipped: no product image"
            elif not os.path.exists(product_image):
                debug_log(f"Product image file not found: {product_image}")
                results["analysis_status"] = "skipped: product image file not found"
    else:
        # Add information about why analysis was skipped
        if not GEMINI_API_KEY:
            debug_log("Skipping Gemini analysis: No API key configured")
            results["analysis_status"] = "skipped: no API key"
        elif product_analyzed:
            debug_log("Skipping Gemini analysis: Already performed")
            results["analysis_status"] = "skipped: already performed"
    
    debug_log("Completed capture_all API call")
    
    # Force garbage collection after completing the analysis
    gc.collect()
    
    return results

@app.get("/last_captured")
async def get_last_captured():
    """Get the last captured images for all cameras."""
    return last_captured_images

@app.get("/last_detection")
async def get_last_detection():
    """Get the last detection results for all cameras."""
    return last_detection_results

@app.get("/best_detection")
async def get_best_detection():
    """Get the detection with highest confidence across all cameras."""
    return best_detection_result or {"message": "No detections available"}

@app.get("/processing_status")
async def get_processing_status():
    """Get the current processing status for all cameras."""
    return processing_status

@app.get("/system_status")
async def get_system_status():
    """Get the system status including memory usage and active clients."""
    import psutil
    
    try:
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage = {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "used_gb": round(memory.used / (1024 ** 3), 2),
            "percent_used": memory.percent
        }
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get active clients count
        clients = {}
        with client_lock:
            for camera_id, count in active_stream_clients.items():
                clients[camera_id] = count
        
        # Get camera frame rates
        camera_status = {}
        for camera_id in camera_streams.keys():
            with frame_locks.get(camera_id, threading.Lock()):
                last_frame_time = frame_timestamps.get(camera_id, 0)
                has_frames = latest_frames.get(camera_id) is not None
            
            # Calculate time since last frame
            current_time = time.time()
            time_since_last_frame = current_time - last_frame_time if last_frame_time > 0 else float('inf')
            
            camera_status[camera_id] = {
                "connected": has_frames,
                "last_frame_time": datetime.fromtimestamp(last_frame_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if last_frame_time > 0 else "Never",
                "seconds_since_last_frame": round(time_since_last_frame, 2),
                "status": "active" if time_since_last_frame < 5 else "stale"
            }
        
        return {
            "memory": memory_usage,
            "cpu_percent": cpu_percent,
            "active_clients": clients,
            "cameras": camera_status,
            "processing_queue_size": gpu_task_queue.qsize(),
            "gpu_worker_running": gpu_worker_running,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/force_gc")
async def force_garbage_collection():
    """Force garbage collection to free up memory."""
    try:
        before = gc.get_count()
        collected = gc.collect(generation=2)
        after = gc.get_count()
        
        return {
            "status": "success",
            "collected_objects": collected,
            "before": before,
            "after": after,
            "message": f"Garbage collection completed. Collected {collected} objects."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/camera_status")
async def get_camera_status():
    """Get status of camera streams including FPS and frame timestamps."""
    status = {}
    
    for camera_id in camera_streams.keys():
        with frame_locks.get(camera_id, threading.Lock()):
            last_frame_time = frame_timestamps.get(camera_id, 0)
            has_frames = latest_frames.get(camera_id) is not None
        
        # Calculate time since last frame
        current_time = time.time()
        time_since_last_frame = current_time - last_frame_time if last_frame_time > 0 else float('inf')
        
        # Get client count
        with client_lock:
            client_count = active_stream_clients.get(camera_id, 0)
        
        status[camera_id] = {
            "connected": has_frames,
            "active_clients": client_count,
            "last_frame_time": datetime.fromtimestamp(last_frame_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if last_frame_time > 0 else "Never",
            "seconds_since_last_frame": round(time_since_last_frame, 2),
            "status": "active" if time_since_last_frame < 5 else "stale"
        }
    
    return status

# Get API key from environment variable
GEMINI_API_KEY = "AIzaSyBPBIHYDsoaQEh75NLz2Hpd4FqH8a9QYps"
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables. Product analysis will not work.")

# Initialize Gemini API and add routes if API key is available
if GEMINI_API_KEY:
    try:
        from gemini_api import add_gemini_routes
        add_gemini_routes(app, GEMINI_API_KEY)
        print("Gemini API integration initialized successfully.")
    except ImportError:
        print("Warning: gemini_api module not found. Skipping Gemini API integration.")
        @app.get("/analyze_product")
        async def analyze_product_placeholder():
            return {"error": "Gemini API integration not available", "message": "Please make sure gemini_api.py is in your project"}
else:
    @app.get("/analyze_product")
    async def analyze_product_placeholder():
        return {"error": "Gemini API key not configured", "message": "Please set GEMINI_API_KEY in environment variables"}
    
    @app.post("/process_capture")
    async def process_capture_placeholder():
        return {"error": "Gemini API key not configured", "message": "Please set GEMINI_API_KEY in environment variables"}

# Clean up function to be called when the application shuts down
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down."""
    global camera_threads_running
    
    print("Shutting down application, cleaning up resources...")
    
    # Stop camera reader threads
    camera_threads_running = False
    time.sleep(1)  # Allow threads to terminate
    
    # Clear large objects from memory
    if yolo_model is not None:
        del yolo_model
    
    # Clear the frame buffers
    for camera_id in latest_frames:
        if latest_frames[camera_id] is not None:
            latest_frames[camera_id] = None
    
    # Run garbage collection to free memory
    gc.collect()
    
    print("Application shutdown complete")

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize resources when the application starts."""
    print("Starting up application, initializing resources...")
    
    try:
        # Start the cleanup thread
        start_cleanup_thread()
        print("File cleanup thread started")
        
        # Ensure all directories exist
        for directory in ["static", "static/captures", "static/detections", "static/product_info"]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize active client counters
        for camera_id in ["camera1", "camera2"]:
            active_stream_clients[camera_id] = 0
        
        # Run garbage collection
        gc.collect()
        
        print("Application startup complete")
    except Exception as e:
        print(f"Error during application startup: {str(e)}")

if __name__ == "__main__":
    # Set up a memory monitoring function that runs periodically
    def memory_monitor():
        while True:
            try:
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent > 90:  # High memory usage
                    print(f"WARNING: High memory usage detected: {memory.percent}%. Forcing garbage collection.")
                    gc.collect()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"Error in memory monitor: {str(e)}")
                time.sleep(60)  # If error, wait longer before retry
    
    # Start memory monitor in a background thread
    threading.Thread(target=memory_monitor, daemon=True).start()
    
    # Start the application
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)