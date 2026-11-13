import unittest

import numpy as np

from src.lane_detector import detect_lanes, region_of_interest

class TestLaneDetector(unittest.TestCase):
    def test_empty_frame_does_not_crash(self):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        overlaid, edges = detect_lanes(frame)
        self.assertEqual(overlaid.shape, frame.shape)
        self.assertEqual(edges.shape, frame.shape[:2])
        self.assertFalse(edges.any())
        
    def test_roi_preserves_shape(self):
        frame = np.ones((540, 960) dtype=np.uint8) * 255
        masked = region_of_interest(frame)
        self.assertEqual(masked.shape, frame.shape)
        self.assertEqual(masked[0, 0], 0)
        self.assertEqual(masked[539, 480], 255)
        
if __name__ == "__main__":
    unittest.main()