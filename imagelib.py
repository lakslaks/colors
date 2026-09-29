import os, re, requests, base64
from urllib.parse import urlparse
import matplotlib.pyplot as plt, numpy as np
from typing import Callable, Dict, Tuple, Any, Optional, Iterable, Union, List, Literal
from html import escape
from IPython.display import HTML, display

import cv2
from deepface import DeepFace
from ultralytics import YOLO

class ImageManager:

    def __init__ (self, model: str = 'yolo26n.pt', **kwargs):
        self.model = YOLO(model)
    
    def load(self, src: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load an image from a NumPy array, a local file path, or an HTTP/HTTPS URL.
    
        Parameters
        ----------
        src : str | np.ndarray
            The image source. Supported forms:
            - A NumPy array representing an image (H×W×C).
            - A string path to a local image file.
            - A string URL pointing to an image (http/https).
    
        Returns
        -------
        np.ndarray
            The loaded image as a BGR NumPy array suitable for OpenCV.
    
        Raises
        ------
        ValueError
            If the input is not a valid image array, file path, or URL.
        """
        if isinstance(src, np.ndarray) and len(src.shape) > 2:
            image = src
        elif isinstance(src, str) and os.path.isfile(src) and os.path.exists(src):
            image = cv2.imread(src)
        elif isinstance(src, str) and urlparse(src).scheme in ("http", "https"):
            response = requests.get(src, stream=True)
            response.raise_for_status()  # Raise error if download fails
            image_data = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        else:
            raise ValueError("Input must be a file path, URL, or image array.")
        return image
        

    def show(self, images, captions=None, title=None, cols=5, img_height=100, img_width=150):
        """
        images:
            - ndarray
            - list of ndarrays
            - dict: {caption: ndarray}
    
        captions:
            - scalar → caption 1, caption 2, ...
            - list/tuple → zip(captions, images)
            - dict → override only when images is NOT a dict
            - None → use auto-generated captions
        """
        if isinstance(images, dict): # Normalize images into dict form
            img_dict = images
        elif isinstance(images, list): # list → auto-generate captions
            img_dict = {f"image {i+1}": img for i, img in enumerate(images)}
    
            # Apply captions ONLY if provided AND images was NOT a dict
            if captions is not None:
                if isinstance(captions, str): # Scalar → caption 1, caption 2, ...
                    img_dict = {f"{captions} {i+1}": img for i, img in enumerate(images)}
                elif isinstance(captions, (list, tuple)): # List/tuple → zip(captions, images)
                    img_dict = {cap: img for cap, img in zip(captions, images)}
                elif isinstance(captions, dict): # Dict → override keys
                    img_dict = {captions.get(old_key, old_key): img for old_key, img in img_dict.items()}
        else: # single image
            img_dict = {"image 1": images}
            if isinstance(captions, str):
                img_dict = {captions: images}
    
        def to_base64(img):
            # Normalize width
            h, w = img.shape[:2]
            scale = img_height / h
            new_h = img_height
            new_w = int(w * scale)            
        
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
            ok, buffer = cv2.imencode(".png", resized)
            if not ok:
                raise ValueError("Failed to encode image")
        
            b64 = base64.b64encode(buffer).decode("utf-8")
            return f'<img src="data:image/png;base64,{b64}" width="{new_w}">'    

        # Build HTML table
        html = ""
        if title:
            html += f"<h3 style='margin-bottom:8px;'>{title}</h3>"
        html += "<table style='border-collapse: collapse;'><tr>"
    
        for caption, img in img_dict.items():
            tag = to_base64(img)
    
            html += ("<td style='padding:8px; border:1px solid #ccc; text-align:center;'>"
                     f"{tag}<br><div style='font-size:12px; color:#555;'>{caption}</div>"
                     "</td>")
        html += "</tr></table>"
    
        display(HTML(html))
    
    def compare_face(self, baseline: Union[np.ndarray, List[np.ndarray]],
        candidates: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        model: str = 'ArcFace', display=False, **kwargs):
    
        if isinstance(baseline, (str, np.ndarray)):
            base, base_list = baseline,[baseline]
        elif isinstance(baseline, (list, tuple)):
            if len(baseline) == 0:
                raise ValueError("Baseline list is empty")
            base, base_list = baseline[0], list(baseline)
        else:
            raise TypeError("baseline must be ndarray or list of ndarrays")
    
        if candidates is None:
            # infer from baseline list
            if len(base_list) > 1:
                candidates = base_list[1:]
            else:
                raise ValueError("No candidates provided and baseline has no extras")
        else:
            if isinstance(candidates, (str, np.ndarray)):
                candidates = [candidates]
            elif isinstance(candidates, (list, tuple)):
                candidates = list(candidates)
            else:
                raise TypeError("candidates must be ndarray or list of ndarrays")                                                                            
    
        results = []
        for candidate in candidates:
            try:
                rs = DeepFace.verify(base, candidate, model_name = model)
                results.append ({'status': 'success', 'verified':rs['verified'], 'confidence': rs['confidence']})
            except Exception as e:
                results.append ({'status': 'error', 'verified':False, 'confidence': 0})
        return results
                
    def profile_photo(self, images, model='opencv', extract_summary=True, seq=0):
        """
        images can be:
          - np.ndarray
          - list/tuple of ndarrays
          - string (path)
          - list/tuple of strings
          - dict: {key: ndarray_or_string, ...}
        """
        def analyze (image, extract = extract_summary):
            _res = DeepFace.analyze(img_path=image, detector_backend=model, enforce_detection=True, silent=True)
            resmap = {'confidence':'face_confidence', 'gender':'dominant_gender',  'race':'dominant_race', 'age':'age', 'emotion':'dominant_emotion'}
            if extract == True:
                _res = {k:_res[seq][v] for k,v in resmap.items()}
            return _res 
        
        if isinstance(images, dict): # dict
            results = {k: analyze(v) for k, v in images.items()}
    
        if isinstance(images, (list, tuple)): # list/tuple
            results = [analyze(item) for item in images]
    
        if isinstance (images, (str, np.ndarray)): # single
            results = analyze(images)
           
        return results