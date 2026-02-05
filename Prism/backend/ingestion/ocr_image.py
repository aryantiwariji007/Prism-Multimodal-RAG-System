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
        if hasattr(self, 'initialized') and self.initialized:
            return
        self.ocr = None
        self.initialized = False

    def _initialize(self):
        if self.initialized:
            return
        
        logger.info("Initializing PaddleOCR...")
        try:
            # Lazy import to avoid DLL load errors on startup
            from paddleocr import PaddleOCR
            
            # In PaddleOCR 3.x, use 'device' instead of 'use_gpu'
            # 'use_angle_cls' is replaced by 'use_textline_orientation'
            # 'enable_mkldnn=False' is crucial for some Windows environments
            self.ocr = PaddleOCR(
                use_textline_orientation=False, 
                lang='en', 
                device='cpu',
                enable_mkldnn=False
            )
            self.initialized = True
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.ocr = None
            self.initialized = True # Mark as tried

    def extract_text(self, image_path: str) -> str:
        """
        Extracts text from an image file using PaddleOCR.
        Returns the combined text content.
        """
        try:
            if not self.initialized:
                self._initialize()

            if not self.ocr:
                logger.warning("PaddleOCR not initialized or failed, skipping text extraction.")
                return ""

            # PaddleOCR 3.x predict() is the preferred way, ocr() is an alias
            result = self.ocr.ocr(image_path)
            if not result or result[0] is None:
                return ""

            extracted_lines = []
            
            # Handle PaddleOCR 3.x / PaddleX format (list of OCRResult/dict)
            if isinstance(result[0], (dict, object)) and hasattr(result[0], 'get') and 'rec_texts' in result[0]:
                for res in result:
                    if 'rec_texts' in res:
                        extracted_lines.extend(res['rec_texts'])
            # Handle PaddleOCR 2.x format: [[[[box], (text, score)], ...]]
            elif isinstance(result[0], list):
                for line in result:
                    for word_info in line:
                        if isinstance(word_info, (list, tuple)) and len(word_info) > 1:
                            text = word_info[1][0]
                            extracted_lines.append(text)
            
            return "\n".join(extracted_lines)

        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return ""

# Global instance
prism_ocr = PrismOCR()
