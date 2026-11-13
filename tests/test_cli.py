"""Tests that don't depend on actual trained weights or a GPU."""

import unittest

from src.predict_yolo import parse_args

class TestCLI(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args(["--source", "example.jpg"])
        self.assertEqual(args.weights, "models/yolo_v1/best.pt")
        self.assertEqual(args.source, "data/images")
        self.assertEqual(args.imgsz, 416)
        self.assertAlmostEqual(args.conf, 0.25)
        self.assertFalse(args.save_txt)
        self.assertFalse(args.save_conf)
        self.assertFalse(args.save_crop)
        self.assertFalse(args.view_img)
        self.assertFalse(args.augment)
        self.assertFalse(args.visualize)
        self.assertEqual(args.update, 0)
        self.assertEqual(args.project, "runs/detect")
        self.assertEqual(args.name, "exp")
        self.assertTrue(args.exist_ok)
    
    def test_predict_custom_settings(self):
        args = parse_args(["--weights", "custom.pt", "--source", "video.mp4", "--imgsz", "320", "--save-txt"])
        self.assertEqual(args.weights, "custom.pt")
        self.assertEqual(args.source, "video.mp4")
        self.assertEqual(args.imgsz, 320)
        self.assertTrue(args.save_txt)
        self.assertAlmostEqual(args.conf, 0.10)
        
if __name__ == "__main__":
    unittest.main()