import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from model import SiameseUNetDamageAssessor
import cv2
import json

class DisasterAssessmentPipeline:
    """
    An inference wrapper for the Siamese U-Net Damage Assessment model.
    Handles image preprocessing, runs the PyTorch model, and returns
    numpy arrays for segmentation and damage maps.
    """
    def __init__(self, model_path=None, device=None, num_disaster_types=10):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SiameseUNetDamageAssessor(num_disaster_types=num_disaster_types)
        
        self.weights_loaded = False
        if model_path:
            # Load weights if a pre-trained path is provided
            # weights_only=True is applied for safe loading
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.weights_loaded = True
            
        self.model.to(self.device)
        self.model.eval()
        
        # Default preprocessing transforms (Resize, to tensor, ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        """Convers inputs to normalized PyTorch tensor ready for inference"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif hasattr(image, 'convert'):
            image = image.convert("RGB")
            
        return self.transform(image).unsqueeze(0).to(self.device)

    def _generate_stats_and_geojson(self, seg_mask, damage_mask):
        stats = {
            'objects_counted': {'No Damage': 0, 'Minor': 0, 'Major': 0, 'Destroyed': 0},
            'impact_area_ha': {'No Damage': 0.0, 'Minor': 0.0, 'Major': 0.0, 'Destroyed': 0.0},
            'impact_area_sq_km': {'No Damage': 0.0, 'Minor': 0.0, 'Major': 0.0, 'Destroyed': 0.0}
        }
        features = []
        gsd_m = 0.3 # 0.3 meters per pixel
        sq_m_per_pixel = gsd_m * gsd_m
        sq_m_to_ha = 0.0001
        sq_m_to_sq_km = 1e-6
        
        # Class 0 = No Damage (building present but undamaged)
        classes = {0: 'No Damage', 1: 'Minor', 2: 'Major', 3: 'Destroyed'}
        for dmg_val, dmg_name in classes.items():
            if dmg_val == 0:
                # No Damage = pixels where building exists (seg_mask==1) but damage_mask==0
                binary_mask = ((seg_mask > 0) & (damage_mask == 0)).astype(np.uint8)
            else:
                binary_mask = (damage_mask == dmg_val).astype(np.uint8)
            
            num_labels, _ = cv2.connectedComponents(binary_mask)
            stats['objects_counted'][dmg_name] = max(0, num_labels - 1)
            
            pixel_count = int(np.sum(binary_mask))
            area_sq_m = pixel_count * sq_m_per_pixel
            stats['impact_area_ha'][dmg_name] = round(float(area_sq_m * sq_m_to_ha), 3)
            stats['impact_area_sq_km'][dmg_name] = round(float(area_sq_m * sq_m_to_sq_km), 4)

            # Contours for GeoJSON
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 10:
                    continue
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                coords = []
                for pt in approx:
                    x, y = int(pt[0][0]), int(pt[0][1])
                    lon = 100.0 + x * 0.0001
                    lat = 20.0 - y * 0.0001
                    coords.append([lon, lat])
                    
                if len(coords) >= 3:
                    coords.append(coords[0])
                    features.append({
                        "type": "Feature",
                        "properties": {"damage": dmg_name},
                        "geometry": {"type": "Polygon", "coordinates": [coords]}
                    })
                    
        geojson_str = json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
        return stats, geojson_str

    def predict(self, pre_image, post_image, disaster_type_id):
        """
        Run inference on pre and post images.
        Returns:
            seg_map (np.ndarray): 2D binary array for building footprint
            damage_map (np.ndarray): 2D Int array for damage class (0=No damage, 1=Minor, 2=Major, 3=Destroyed)
        """
        if not self.weights_loaded:
            # If no trained weights are loaded, generate a realistic MOCK prediction for the UI demo.
            if isinstance(pre_image, str):
                pre_img = Image.open(pre_image).convert("L").resize((256, 256))
            elif hasattr(pre_image, 'resize'):
                pre_img = pre_image.resize((256, 256)).convert("L")
            else:
                pre_img = Image.fromarray(np.array(pre_image)).convert("L").resize((256, 256))

            if isinstance(post_image, str):
                base_img = Image.open(post_image).convert("L").resize((256, 256))
            elif hasattr(post_image, 'resize'):
                base_img = post_image.resize((256, 256)).convert("L")
            else:
                base_img = Image.fromarray(np.array(post_image)).convert("L").resize((256, 256))
                
            pre_gray = np.array(pre_img).astype(np.float32)
            post_gray = np.array(base_img).astype(np.float32)
            
            # Use visual difference to map accurate severity classes instead of random noise
            diff = np.abs(pre_gray - post_gray)
            diff_norm = np.clip(diff / 255.0, 0, 1)
            
            # Assume any non-black pixel in pre-image constitutes active area (segmentation)
            seg_mask = (pre_gray > 20).astype(np.float32)
            
            random_damage = np.zeros((256, 256), dtype=np.float32)
            # Assign damage based on actual visual alteration (helps demo mode align perfectly to image)
            random_damage[(diff_norm > 0.15) & (diff_norm <= 0.30)] = 1 # Minor
            random_damage[(diff_norm > 0.30) & (diff_norm <= 0.50)] = 2 # Major
            random_damage[diff_norm > 0.50] = 3                         # Destroyed
            
            damage_map_masked = random_damage * seg_mask
            
            # Spatial Smoothing Filter (acts like a CRF to remove noise)
            kernel = np.ones((3, 3), np.uint8)
            damage_map_masked = cv2.morphologyEx(damage_map_masked.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            stats, geojson_str = self._generate_stats_and_geojson(seg_mask, damage_map_masked)
            return seg_mask, damage_map_masked, stats, geojson_str

        pre_tensor = self.preprocess_image(pre_image)
        post_tensor = self.preprocess_image(post_image)
        
        # Disaster type tag parameter tensor
        tag_tensor = torch.tensor([disaster_type_id], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            seg_logits, damage_logits = self.model(pre_tensor, post_tensor, tag_tensor)
            
            # Post-process segmentation (Sigmoid thresholding at 0.5)
            seg_probs = torch.sigmoid(seg_logits)
            seg_mask = (seg_probs > 0.5).float()
            
            # Post-process damage (Softmax and argmax)
            damage_probs = F.softmax(damage_logits, dim=1)
            damage_mask = torch.argmax(damage_probs, dim=1).float()
            
            # Contextual Check: Mask the damage prediction with the building footprint
            damage_map_masked = damage_mask * seg_mask.squeeze(1)
            
        final_seg = seg_mask.cpu().numpy()[0, 0]
        final_damage = damage_map_masked.cpu().numpy()[0]
        
        # Spatial Smoothing Filter (acts like a CRF to remove noise)
        kernel = np.ones((3, 3), np.uint8)
        final_damage = cv2.morphologyEx(final_damage.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        stats, geojson_str = self._generate_stats_and_geojson(final_seg, final_damage)
        return final_seg, final_damage, stats, geojson_str


def calculate_iou(pred_mask, true_mask, num_classes=4):
    """
    Calculates the Intersection over Union (IoU) per class and the Mean IoU.
    
    Args:
        pred_mask (np.array): Predicted multi-class mask [H, W]
        true_mask (np.array): Ground truth mask [H, W]
        num_classes (int): Number of target classes
        
    Returns:
        dict: Dictionary containing IoU for each class and the Mean IoU.
    """
    iou_dict = {}
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        
        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()
        
        if union == 0:
            iou = float('nan') # Class not present in prediction nor target
        else:
            iou = intersection / union
            ious.append(iou)
            
        iou_dict[f'Class_{cls}_IoU'] = iou
        
    iou_dict['Mean_IoU'] = np.nanmean(ious) if len(ious) > 0 else float('nan')
    return iou_dict
