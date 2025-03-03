import time
import os
import cv2
import requests
import numpy as np
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import tempfile
import threading
from queue import Queue

_browser_instance = None
_browser_lock = threading.Lock()


def get_browser_instance():
    """Returns a singleton browser instance that can be reused across calls"""
    global _browser_instance
    
    with _browser_lock:
        if _browser_instance is None or not is_browser_alive(_browser_instance):
            # Configure browser options
            options = webdriver.EdgeOptions()
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--start-maximized")
            options.add_argument("--log-level=3")
            options.add_argument("--headless")
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0")
            
            print("Starting new Edge WebDriver instance...")
            try:
                _browser_instance = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
            except Exception as e:
                print(f"Error creating browser instance: {e}")
                raise
                
            # Navigate to Google Images once to initialize
            try:
                _browser_instance.get("https://images.google.com/")
                WebDriverWait(_browser_instance, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                print("Browser initialized with Google Images")
            except Exception as e:
                print(f"Error initializing browser: {e}")
                
    return _browser_instance

def is_browser_alive(driver):
    """Check if the browser session is still valid"""
    if driver is None:
        return False
    try:
        # Try to access a property to see if session is still alive
        _ = driver.title
        return True
    except:
        return False

def close_browser_instance():
    """Close the browser instance when it's no longer needed"""
    global _browser_instance
    with _browser_lock:
        if _browser_instance is not None:
            try:
                _browser_instance.quit()
            except:
                pass
            _browser_instance = None


def get_edge_driver():
    """Get Edge WebDriver with proper executable path handling for PyInstaller."""
    options = webdriver.EdgeOptions()
    options.add_argument("--headless")
    # Other options as before
    
    try:
        # First try: Use bundled driver if available (when packaged as EXE)
        if getattr(sys, 'frozen', False):
            # Running as compiled EXE
            base_path = os.path.dirname(sys.executable)
            driver_path = os.path.join(base_path, "msedgedriver.exe")
            if os.path.exists(driver_path):
                return webdriver.Edge(service=Service(driver_path), options=options)
        
        # Second try: Use webdriver_manager to download if needed
        return webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
    except Exception as e:
        print(f"Error setting up Edge driver: {e}")
        raise

def google_lens_search(image_path):
    """Uses a shared browser instance to upload an image to Google Lens and fetch search results."""
    try:
        # Get the shared browser instance instead of creating a new one each time
        driver = get_browser_instance()
        
        # Reset to Google Images search page
        driver.get("https://images.google.com/")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        print("Opened Google Images...")
        time.sleep(2)
        
        # Click on camera icon for image search (more reliable than Google Lens)
        print("Looking for camera icon...")
        # Try multiple different selectors for the camera button
        camera_selectors = [
            "div[aria-label='Search by image']",
            "div[jsname='F78hdc']",
            "div[data-ved] span.ZaIChf",
            "div.nDcEnd",
            "div.xSQxL"  # Current selector as of Feb 2025
        ]
        
        camera_found = False
        for selector in camera_selectors:
            try:
                camera_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                camera_button.click()
                camera_found = True
                print(f"Found camera button with selector: {selector}")
                break
            except:
                continue
                
        if not camera_found:
            # Try with XPath as a last resort
            try:
                camera_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[@aria-label='Search by image']"))
                )
                camera_button.click()
                camera_found = True
            except:
                print("Failed to find camera button with any selector")
                driver.save_screenshot("no_camera_icon.png")
                # Don't quit the driver here, just return empty results
                return []
        
        print("Waiting for upload options...")
        time.sleep(2)
        
        # Look for the upload image option
        upload_tabs = driver.find_elements(By.XPATH, "//div[text()='Upload an image']")
        if upload_tabs:
            upload_tabs[0].click()
            print("Clicked on 'Upload an image' tab")
            
        # Look for exact upload button if needed
        try:
            upload_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, 
                    "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div[2]/c-wiz/div[2]/div/div[2]/div[2]/div/div[2]/span"))
            )
            upload_button.click()
            print("Clicked on upload button using exact XPath")
        except:
            print("Could not find exact upload button - continuing with file input approach")
        
        # Now look for the file input element with multiple approaches
        file_input = None
        selectors = [
            "input[type='file']",
            "input#awyMjb",  # Specific Google input ID
            "input.cB9M7"    # Another common class
        ]
        
        for selector in selectors:
            try:
                file_input = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                if file_input:
                    break
            except:
                continue
        
        if not file_input:
            print("Could not find file input element")
            driver.save_screenshot("no_file_input.png")
            return []
        
        # Get absolute path of the image
        abs_path = os.path.abspath(image_path)
        print(f"Uploading image: {abs_path}")
        
        # Upload the image
        file_input.send_keys(abs_path)
        
        # Wait for results to load
        print("Waiting for results to load...")
        time.sleep(15)  # Increased wait time
        
        # Take screenshot to see what we're working with
        driver.save_screenshot("after_upload.png")
        print("Screenshot saved as after_upload.png")
        
        # Wait for search results to appear
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".g-blk"))
            )
            print("Search results page loaded")
        except:
            print("Timeout waiting for search results page - continuing anyway")
            
        # Extract image search results using JavaScript
        print("Using JavaScript to extract images...")
        try:
            # Get all image elements with their sources and parent links
            similar_images = driver.execute_script("""
                const images = Array.from(document.querySelectorAll('img'))
                    .filter(img => img.src && img.src.startsWith('http') && 
                                  img.width > 60 && img.height > 60);
                    
                return images.map(img => {
                    let parent = img;
                    // Try to find the closest anchor parent
                    while (parent && parent.tagName !== 'A' && parent.parentElement) {
                        parent = parent.parentElement;
                    }
                    
                    return {
                        src: img.src,
                        link: (parent && parent.tagName === 'A') ? parent.href : null,
                        width: img.width,
                        height: img.height
                    };
                }).filter(item => item.src);
            """)
            
            print(f"Found {len(similar_images)} images via JavaScript")
            
            # Process the JavaScript results
            image_urls = []
            result_links = []
            
            for item in similar_images[:10]:  # Process more results to ensure we get enough
                if isinstance(item, dict):
                    src = item.get('src')
                    link = item.get('link')
                    
                    if src and src.startswith('http') and not src.endswith('.gif'):
                        image_urls.append(src)
                        
                    if link and link.startswith('http'):
                        result_links.append(link)
            
            # Ensure we have at least 5 results if possible
            image_urls = image_urls[:5]
            result_links = result_links[:5]
            
        except Exception as js_error:
            print(f"JavaScript extraction failed: {js_error}")
            
            # Fallback to regular element extraction if JS approach fails
            print("Falling back to regular element extraction...")
            
            # Try multiple selectors for images
            image_selectors = [
                "img.Q4LuWd",     # Modern Google Images thumbnails
                "img.rg_i",       # Traditional Google results
                ".isv-r img",     # Image container images
                "a.wXeWr img",    # Another pattern
                "g-img img"       # Generic Google images
            ]
            
            similar_images = []
            for selector in image_selectors:
                try:
                    similar_images = driver.find_elements(By.CSS_SELECTOR, selector)
                    if similar_images:
                        print(f"Found {len(similar_images)} images with selector: {selector}")
                        break
                except Exception as sel_error:
                    print(f"Selector {selector} failed: {sel_error}")
            
            # Similarly for links
            link_selectors = [
                ".isv-r a",        # Image container links
                "a.VFACy",         # Modern source links
                "a.wXeWr",         # Another common pattern
                "a[jsname='sTFXNd']" # Image result links
            ]
            
            source_links = []
            for selector in link_selectors:
                try:
                    source_links = driver.find_elements(By.CSS_SELECTOR, selector)
                    if source_links:
                        print(f"Found {len(source_links)} links with selector: {selector}")
                        break
                except Exception as sel_error:
                    print(f"Selector {selector} failed: {sel_error}")
            
            # Process results from element extraction
            image_urls = []
            result_links = []
            
            for img in similar_images[:5]:
                try:
                    url = img.get_attribute("src")
                    if url and url.startswith("http"):
                        image_urls.append(url)
                except:
                    continue
                    
            for link in source_links[:5]:
                try:
                    href = link.get_attribute("href")
                    if href and href.startswith("http"):
                        result_links.append(href)
                except:
                    continue
        
        # If we have no image URLs, try one last desperate approach
        if not image_urls:
            print("No images found - trying last resort approach...")
            try:
                # Save the entire page source for debugging
                with open("page_source.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                    
                # Try to get all image elements
                all_images = driver.find_elements(By.TAG_NAME, "img")
                print(f"Found {len(all_images)} total images on page")
                
                # Filter for reasonable sized images
                for img in all_images:
                    try:
                        width = img.get_attribute("width")
                        height = img.get_attribute("height")
                        src = img.get_attribute("src")
                        
                        if src and src.startswith("http") and width and height:
                            if int(width) > 60 and int(height) > 60:
                                image_urls.append(src)
                                
                                # Try to get parent link
                                parent = driver.execute_script("return arguments[0].closest('a')", img)
                                if parent:
                                    href = parent.get_attribute("href")
                                    if href and href.startswith("http"):
                                        result_links.append(href)
                                else:
                                    # Use image source as fallback
                                    result_links.append(src)
                    except:
                        continue
                        
                # Limit to 5 results
                image_urls = image_urls[:5]
                result_links = result_links[:5]
            except Exception as last_error:
                print(f"Last resort approach failed: {last_error}")
        
        print(f"Final results: {len(image_urls)} image URLs and {len(result_links)} source links")
        
        # If we have images but no links, or vice versa, create fallback values
        if image_urls and not result_links:
            result_links = image_urls.copy()
        elif result_links and not image_urls:
            # We need images for visual comparison, so we can't proceed without them
            print("No image URLs found - cannot perform visual comparison")
            return []
            
        # Combine results, ensuring both lists have the same length
        min_length = min(len(image_urls), len(result_links))
        combined_results = []
        
        for i in range(min_length):
            combined_results.append((image_urls[i], result_links[i]))
            
        # Debug: download first few images to verify them
        for i, url in enumerate(image_urls[:3]):
            try:
                response = requests.get(url)
                with open(f"result_image_{i}.jpg", "wb") as f:
                    f.write(response.content)
                print(f"Saved result_image_{i}.jpg")
            except Exception as save_error:
                print(f"Failed to save result image {i}: {save_error}")
                
        return combined_results

    except Exception as e:
        print(f"Error in Selenium: {e}")
        import traceback
        traceback.print_exc()
        
        # Take screenshot of error state
        try:
            driver.save_screenshot("error_state.png")
            print("Error state screenshot saved as error_state.png")
        except:
            pass
            
        # Don't quit the driver here, just return empty results
        return []


def download_image(url):
    """Downloads an image from a URL and converts it to an OpenCV format with support for AVIF/WebP."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            try:
                # First attempt: Try to open directly with PIL/OpenCV
                img = Image.open(io.BytesIO(response.content))
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                print(f"Successfully loaded image using PIL/OpenCV")
                return img_cv
            except Exception as format_error:
                print(f"Error processing image format: {format_error}")
                
                # Second attempt: Save to temporary file and load with OpenCV
                temp_file = f"temp_download_{int(time.time())}.jpg"
                with open(temp_file, "wb") as f:
                    f.write(response.content)
                
                try:
                    # Try to load with OpenCV
                    img_cv = cv2.imread(temp_file)
                    if img_cv is not None:
                        # Clean up temp file and return image
                        os.remove(temp_file)
                        print(f"Successfully loaded image using temp file")
                        return img_cv
                        
                    # If OpenCV can't read it directly, try with browser rendering
                    print("OpenCV couldn't read the image directly, trying with browser rendering")
                    rendered_img = render_image_with_browser(url)
                    if rendered_img is not None:
                        # Remove the temp file
                        os.remove(temp_file)
                        return rendered_img
                    else:
                        # If that also fails, try one last approach with the temp file
                        print("Browser rendering failed, trying to convert format with PIL")
                        try:
                            with Image.open(temp_file) as img:
                                # Convert to a simple format (PNG) and save
                                simple_format = "temp_converted.png"
                                img.save(simple_format)
                                
                                # Try reading with OpenCV
                                img_cv = cv2.imread(simple_format)
                                
                                # Clean up
                                os.remove(simple_format)
                                os.remove(temp_file)
                                
                                if img_cv is not None:
                                    print("Successfully loaded after format conversion")
                                    return img_cv
                        except:
                            # Clean up if still exists
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                except Exception as e:
                    # Clean up if still exists
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    print(f"Error in second attempt: {e}")
                    
        else:
            print(f"Failed to download image, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {e}")
    return None

def render_image_with_browser(url):
    """Use the shared browser instance to render images in formats like AVIF/WebP."""
    try:
        # Get shared browser instance
        driver = get_browser_instance()
        
        # Create a simple HTML page that displays just the image
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; }}
                img {{ max-width: 100%; max-height: 100vh; object-fit: contain; }}
            </style>
        </head>
        <body>
            <img src="{url}" alt="Image">
        </body>
        </html>
        """
        
        # Create a temporary HTML file
        temp_html = f"temp_page_{int(time.time())}.html"
        with open(temp_html, "w") as f:
            f.write(html_content)
            
        # Load the page and wait for image to render
        driver.get(f"file:///{os.path.abspath(temp_html)}")
        time.sleep(2)  # Wait for image to load
        
        # Take screenshot
        temp_screenshot = f"temp_screenshot_{int(time.time())}.png"
        driver.save_screenshot(temp_screenshot)
        
        # Load the screenshot with OpenCV
        img_cv = cv2.imread(temp_screenshot)
        
        # Clean up temporary files
        os.remove(temp_html)
        os.remove(temp_screenshot)
        
        return img_cv
        
    except Exception as e:
        print(f"Error rendering image with browser: {e}")
        return None

# ResNet model for feature extraction - load once and reuse
_resnet_model = None
_resnet_preprocess = None

def get_resnet_model():
    """Lazily load the ResNet model when needed."""
    global _resnet_model, _resnet_preprocess
    
    if _resnet_model is None:
        print("Loading ResNet model...")
        _resnet_model = models.resnet18(pretrained=True)
        # Remove the classification layer
        _resnet_model = torch.nn.Sequential(*(list(_resnet_model.children())[:-1]))
        _resnet_model.eval()
        
        # Define image transformations
        _resnet_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    return _resnet_model, _resnet_preprocess

def extract_features_resnet(image, device="cpu"):
    """Extract features using pre-trained ResNet model with optional GPU support."""
    try:
        model, preprocess = get_resnet_model()
        
        # Move model to specified device if needed
        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)
        
        # Process the image
        if isinstance(image, str):
            # Load image from path
            img = cv2.imread(image)
            if img is None:
                print(f"Could not load image: {image}")
                return None
        else:
            img = image.copy()
            
        # Convert BGR to RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Apply transformations
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Move tensor to device if using GPU
        if device == "cuda" and torch.cuda.is_available():
            img_tensor = img_tensor.to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(img_tensor)
            
        # Convert to numpy array (moving back to CPU if needed)
        if device == "cuda" and torch.cuda.is_available():
            features = features.cpu().numpy().flatten()
        else:
            features = features.numpy().flatten()
        
        return features
    except Exception as e:
        print(f"Error extracting ResNet features: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_similarity_deep_features(features1, features2):
    """Compute similarity between two feature vectors using cosine similarity."""
    try:
        if features1 is None or features2 is None:
            print("One of the feature vectors is None")
            return 0
            
        # Reshape features if needed
        if features1.ndim == 1:
            features1 = features1.reshape(1, -1)
        if features2.ndim == 1:
            features2 = features2.reshape(1, -1)
            
        # Calculate cosine similarity
        sim = cosine_similarity(features1, features2)[0][0]
        # Convert to percentage (cosine similarity ranges from -1 to 1)
        return ((sim + 1) / 2) * 100
    except Exception as e:
        print(f"Error computing deep feature similarity: {e}")
        return 0

def compute_similarity(image1, image2):
    """Computes combined similarity between two images using multiple metrics."""
    try:
        # If either image is None, return 0
        if image1 is None or image2 is None:
            print("One of the images is None")
            return 0
            
        # First try the deep learning approach
        try:
            print("Computing deep learning feature similarity...")
            features1 = extract_features_resnet(image1)
            features2 = extract_features_resnet(image2)
            
            deep_similarity = compute_similarity_deep_features(features1, features2)
            print(f"Deep learning similarity: {deep_similarity:.2f}%")
            
            # If we got a valid result from the deep learning approach, return it
            if deep_similarity > 0:
                return deep_similarity
            
            # Otherwise, fall back to traditional methods
            print("Falling back to traditional image similarity methods")
        except Exception as deep_error:
            print(f"Error with deep learning similarity: {deep_error}")
            print("Falling back to traditional image similarity methods")
            
        # Handle images with different number of channels (e.g., grayscale vs color)
        if len(image1.shape) != len(image2.shape):
            # Convert both to grayscale for consistent comparison
            if len(image1.shape) == 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # Ensure images are the same size for comparison
        h1, w1 = image1.shape[:2]
        image2 = cv2.resize(image2, (w1, h1))
        
        # Convert to grayscale for SSIM if they're color images
        if len(image1.shape) == 3:
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            image1_gray = image1
            image2_gray = image2
        
        # Calculate SSIM
        similarity, _ = ssim(image1_gray, image2_gray, full=True)
        
        # For color images, also calculate histogram comparison
        if len(image1.shape) == 3 and len(image2.shape) == 3:
            # Calculate histogram comparison
            hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Compare histograms
            hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Combine the metrics with weight towards SSIM
            combined_similarity = (similarity * 0.7) + (hist_similarity * 0.3)
        else:
            # For grayscale, just use SSIM
            combined_similarity = similarity
            
        return combined_similarity * 100  # Convert to percentage
    except Exception as e:
        print(f"Error computing similarity: {e}")
        import traceback
        traceback.print_exc()
        return 0  # Return 0% similarity on error

def check_filename_copyright(image_path, source_url):
    """Checks if the image file name matches parts of the source link."""
    try:
        filename = os.path.basename(image_path).split('.')[0].lower()  # Extract file name without extension
        domain = urlparse(source_url).netloc.lower()  # Extract domain from URL
        path = urlparse(source_url).path.lower()  # Extract path from URL
        
        # Check if filename is in domain or path
        return filename in domain or filename in path
    except Exception as e:
        print(f"Error checking filename copyright: {e}")
        return False

def detect_copyright(image_path, use_gpu=False, return_confidence=False):
    """Main function to detect copyright using deep learning features."""
    # Create a directory for results
    temp_dir = tempfile.gettempdir()
    results_dir = os.path.join(temp_dir, "copyright_check_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        if return_confidence:
            return {"result": "Error: Image file not found", "confidence": 0}
        return "Error: Image file not found"
        
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image '{image_path}'")
        if return_confidence:
            return {"result": "Error: Could not load image", "confidence": 0}
        return "Error: Could not load image"
    
    # Save a copy of the original image for reference
    original_filename = os.path.basename(image_path)
    cv2.imwrite(f"{results_dir}/original_{original_filename}", original_image)
    
    print(f"Image loaded successfully: {image_path}")
    print(f"Image size: {original_image.shape}")
    
    # Pre-extract features from the original image - use GPU if available and requested
    print("Extracting features from original image...")
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    original_features = extract_features_resnet(original_image, device=device)
       
    # Step 1: Reverse Image Search
    search_results = google_lens_search(image_path)
    
    if not search_results:
        print("No search results found - Unable to determine copyright status")
        if return_confidence:
            return {"result": "Inconclusive: No search results found", "confidence": 0.3}
        return "Inconclusive: No search results found"
    
    # Step 2: Process search results
    similarities = []
    
    for i, (image_url, source_link) in enumerate(search_results):
        print(f"Processing result {i+1}/{len(search_results)}: {image_url}")
        downloaded_image = download_image(image_url)
        if downloaded_image is None:
            print(f"Failed to download image from {image_url}")
            continue
        
        # Save the downloaded image for reference
        cv2.imwrite(f"{results_dir}/comparison_{i}_{original_filename}", downloaded_image)
        
        # Compute similarity
        similarity_score = compute_similarity(original_image, downloaded_image)
        
        # Save for later reference
        similarities.append({
            'similarity': similarity_score,
            'source': source_link,
            'image_url': image_url
        })
        
        print(f"Similarity with {image_url}: {similarity_score:.2f}%")
        
        # Check filename copyright if similarity is high
        if similarity_score > 90:
            if check_filename_copyright(image_path, source_link):
                print(f"✅ Not Copyrighted (file name found in source URL: {source_link})")
                if return_confidence:
                    return {"result": "No Copyright Issue", "confidence": 0.95}
                return "No Copyright Issue"
    
    # If we have any results with high similarity, consider it copyrighted
    if similarities:
        # Sort by similarity score
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        highest_similarity = similarities[0]['similarity']
        highest_source = similarities[0]['source']
        
        if highest_similarity > 90:
            print(f"🚨 Copyrighted Content Detected! Similarity: {highest_similarity:.2f}%, Source: {highest_source}")
            if return_confidence:
                return {"result": "Copyright Issue Detected", "confidence": highest_similarity / 100}
            return "Copyright Issue Detected"
        elif highest_similarity > 85:
            print(f"⚠️ Possible Copyright Issue! Similarity: {highest_similarity:.2f}%, Source: {highest_source}")
            if return_confidence:
                return {"result": "Possible Copyright Issue", "confidence": highest_similarity / 100}
            return "Possible Copyright Issue"
    
    print("❌ No Copyright Issue Found (Low Similarity)")
    if return_confidence:
        # Return confidence based on highest similarity if available
        confidence = max([s['similarity'] for s in similarities], default=20) / 100 if similarities else 0.2
        return {"result": "No Copyright Issue", "confidence": confidence}
    return "No Copyright Issue"


def batch_copyright_detection(image_paths, use_gpu=False, batch_size=10):
    """
    Process multiple images efficiently by reusing the browser session.
    
    Parameters:
        image_paths (list): List of paths to images to check
        use_gpu (bool): Whether to use GPU acceleration
        batch_size (int): How many images to process before refreshing the browser
    
    Returns:
        dict: Dictionary mapping image paths to copyright detection results
    """
    results = {}
    
    try:
        # Make sure we have a fresh browser instance at the start
        close_browser_instance()
        get_browser_instance()
        
        # Process images in batches
        for i, path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {path}")
            
            # Detect copyright for this image
            result = detect_copyright(path, use_gpu=use_gpu)
            results[path] = result
            
            # Every batch_size images, refresh the browser to prevent memory issues
            if (i + 1) % batch_size == 0 and i < len(image_paths) - 1:
                print(f"Refreshing browser after {batch_size} images...")
                close_browser_instance()
                time.sleep(1)
                get_browser_instance()
    
    finally:
        # Clean up the browser when done
        close_browser_instance()
    
    return results


# # Example usage
# if __name__ == "__main__":
#     image_file = "./key_frames/frame_670.jpg"  # Replace with your image path
#     result = detect_copyright(image_file)
#     print(f"Final decision: {result}")