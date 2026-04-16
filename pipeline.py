import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from model import SiameseUNetDamageAssessor

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
        
        if model_path:
            # Load weights if a pre-trained path is provided
            # weights_only=True is applied for safe loading
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            
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
            
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, pre_image, post_image, disaster_type_id):
        """
        Run inference on pre and post images.
        Returns:
            seg_map (np.ndarray): 2D binary array for building footprint
            damage_map (np.ndarray): 2D Int array for damage class (0=No damage, 1=Minor, 2=Major, 3=Destroyed)
        """
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
            # If there's no building (seg_mask == 0), damage class should be ignored or map to background.
            damage_map_masked = damage_mask * seg_mask.squeeze(1)
            
        return seg_mask.cpu().numpy()[0, 0], damage_map_masked.cpu().numpy()[0]


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
