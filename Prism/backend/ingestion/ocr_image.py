import os
# Disable MKLDNN to avoid "ConvertPirAttribute2RuntimeAttribute" error on some Windows envs
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_mkldnn_deterministic"] = "1"
import logging
import numpy as np
import paddle
try:
    paddle.set_flags({'FLAGS_use_mkldnn': False})
    # Also try this for newer paddle versions
    import paddle.fluid as fluid
    fluid.core.globals()['FLAGS_use_mkldnn'] = False
except Exception:
    pass

# from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class PrismOCR:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized"):
            return
        self.initialized = True
        logger.info("Initializing PaddleOCR...")
        try:
            # Lazy import to avoid DLL load errors on startup if dependencies are broken
            from paddleocr import PaddleOCR
            
            # use_angle_cls=True enables orientation classification
            # lang='en' by default, can be made configurable
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.ocr = None
            # Do not raise here, allow app to start without OCR
            # raise e

    def extract_text(self, image_path: str) -> str:
        """
        Extracts text from an image file using PaddleOCR.
        Returns the combined text content.
        """
        try:
            if not getattr(self, 'ocr', None):
                logger.warning("PaddleOCR not initialized, skipping text extraction.")
                return ""

            result = self.ocr.ocr(image_path)
            if not result or result[0] is None:
                return ""

            # result structure: [[[[x1,y1],[x2,y2],...], ("text", confidence)], ...]
            # We just want the text for now
            extracted_lines = []
            for line in result:
                for word_info in line:
                    text = word_info[1][0]
                    extracted_lines.append(text)
            
            return "\n".join(extracted_lines)

        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return ""

# Global instance
prism_ocr = PrismOCR()
