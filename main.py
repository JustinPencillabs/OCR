import cv2
import numpy as np
import pytesseract
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
EDGE_THRESHOLD1 = 50
EDGE_THRESHOLD2 = 150
DILATION_ITERATIONS = 1
BLUR_RADIUS = 11
MIN_CONTOUR_AREA = 100

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

class CardTextDetector:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load an image from the specified file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not open or find the image: {image_path}")
        return image

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """Convert the image to grayscale and remove noise."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    @staticmethod
    def detect_edges(denoised_gray: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection."""
        return cv2.Canny(denoised_gray, EDGE_THRESHOLD1, EDGE_THRESHOLD2)

    @staticmethod
    def find_largest_contour(edges: np.ndarray) -> np.ndarray:
        """Find the largest contour in the edge-detected image."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def crop_image_to_contour(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Crop the image to the bounding rectangle of the given contour."""
        x, y, w, h = cv2.boundingRect(contour)
        return image[y:y+h, x:x+w]

    @staticmethod
    def dilate_edges(cropped_edges: np.ndarray) -> np.ndarray:
        """Dilate the edges to enhance text regions."""
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(cropped_edges, kernel, iterations=DILATION_ITERATIONS)

    @staticmethod
    def apply_heatmap(dilated_edges: np.ndarray) -> np.ndarray:
        """Apply a heatmap to visualize text-like areas."""
        heatmap = cv2.applyColorMap(dilated_edges, cv2.COLORMAP_JET)
        return cv2.GaussianBlur(heatmap, (BLUR_RADIUS | 1, BLUR_RADIUS | 1), 0)

    @staticmethod
    def extract_text_from_bounding_boxes(cropped_card: np.ndarray, binary_mask: np.ndarray) -> List[str]:
        """Extract text from the bounding boxes around detected text areas."""
        ocr_results = []
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                text_area = cropped_card[y:y+h, x:x+w]
                text = pytesseract.image_to_string(text_area).strip()
                ocr_results.append(text)
                cv2.rectangle(cropped_card, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return ocr_results

    @classmethod
    def detect_card_and_text(cls, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Main function to detect the card and perform OCR on detected text."""
        image = cls.load_image(image_path)
        denoised_gray = cls.preprocess_image(image)
        edges = cls.detect_edges(denoised_gray)
        largest_contour = cls.find_largest_contour(edges)
        cropped_card = cls.crop_image_to_contour(image, largest_contour)

        # Process cropped card for text detection
        gray_cropped = cls.preprocess_image(cropped_card)
        cropped_edges = cls.detect_edges(gray_cropped)
        dilated_edges = cls.dilate_edges(cropped_edges)
        heatmap = cls.apply_heatmap(dilated_edges)
        
        # Create a binary mask for text detection
        _, binary_mask = cv2.threshold(cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract text from bounding boxes
        ocr_results = cls.extract_text_from_bounding_boxes(cropped_card, binary_mask)

        return cropped_card, heatmap, binary_mask, ocr_results

def display_results(cropped_card: np.ndarray, heatmap: np.ndarray, binary_mask: np.ndarray):
    """Display the results of the card text detection."""
    cv2.imshow('Cropped Card with Text Bounding Boxes', cropped_card)
    cv2.imshow('Blurred Heatmap', heatmap)
    cv2.imshow('Binary Mask', binary_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_results(cropped_card: np.ndarray, heatmap: np.ndarray, binary_mask: np.ndarray):
    """Save the result images."""
    cv2.imwrite('cropped_card_with_text.jpg', cropped_card)
    cv2.imwrite('text_heatmap.jpg', heatmap)
    cv2.imwrite('binary_mask.jpg', binary_mask)

def main():
    """Main execution point for the script."""
    image_path = 'card.jpg'
    try:
        detector = CardTextDetector()
        cropped_card, heatmap, binary_mask, ocr_results = detector.detect_card_and_text(image_path)

        display_results(cropped_card, heatmap, binary_mask)
        save_results(cropped_card, heatmap, binary_mask)

        logger.info("OCR Results:")
        for text in ocr_results:
            logger.info(text)

    except FileNotFoundError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
