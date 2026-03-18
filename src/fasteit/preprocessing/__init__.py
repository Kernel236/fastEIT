"""
EIT signal preprocessing pipeline.

- filtering       — cardiac artifact removal (Butterworth, MDN)
- lung_mask       — ventilated pixel identification
- breath_detection — inspiratory/expiratory peak detection
"""
