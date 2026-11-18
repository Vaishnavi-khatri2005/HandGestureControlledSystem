"""
Unit tests for the GestureRecognizer module.

Tests all 7 required gestures individually using geometric landmark models:
1. Open Palm
2. Closed Fist
3. Thumbs Up
4. Thumbs Down
5. Index Finger Up
6. Victory Sign
7. Pinch Gesture
"""

import unittest
from typing import List, Tuple
from src.gesture_recognizer import GestureRecognizer, GestureType


def make_hand(
    thumb_ext: bool = False,
    thumb_dir: str = "neutral",  # "neutral", "up", "down"
    index_ext: bool = False,
    middle_ext: bool = False,
    ring_ext: bool = False,
    pinky_ext: bool = False,
    pinch: bool = False
) -> List[Tuple[int, int, int]]:
    """
    Synthesize realistic 21 landmark coordinates for a right hand.
    Wrist is positioned at (300, 400). Palm scale ~ 100 pixels.
    """
    # 0: Wrist
    lms = [(0, 300, 400)]

    # Thumb: 1: CMC, 2: MCP, 3: IP, 4: TIP
    if thumb_dir == "up":
        lms.extend([(1, 270, 380), (2, 250, 350), (3, 240, 310), (4, 235, 260)])
    elif thumb_dir == "down":
        lms.extend([(1, 270, 390), (2, 250, 410), (3, 240, 440), (4, 235, 480)])
    elif thumb_ext:
        lms.extend([(1, 260, 380), (2, 230, 360), (3, 200, 340), (4, 170, 320)])
    else:
        # Folded thumb
        lms.extend([(1, 280, 380), (2, 270, 360), (3, 280, 350), (4, 290, 345)])

    # Index: 5: MCP, 6: PIP, 7: DIP, 8: TIP
    mcp_x, mcp_y = 270, 300
    if index_ext:
        lms.extend([(5, mcp_x, mcp_y), (6, mcp_x - 5, 240), (7, mcp_x - 10, 190), (8, mcp_x - 15, 140)])
    else:
        lms.extend([(5, mcp_x, mcp_y), (6, mcp_x, 320), (7, mcp_x + 5, 340), (8, mcp_x + 5, 355)])

    # Middle: 9: MCP, 10: PIP, 11: DIP, 12: TIP
    mcp_x, mcp_y = 300, 295
    if middle_ext:
        lms.extend([(9, mcp_x, mcp_y), (10, mcp_x, 230), (11, mcp_x, 180), (12, mcp_x, 130)])
    else:
        lms.extend([(9, mcp_x, mcp_y), (10, mcp_x, 320), (11, mcp_x, 340), (12, mcp_x, 360)])

    # Ring: 13: MCP, 14: PIP, 15: DIP, 16: TIP
    mcp_x, mcp_y = 330, 305
    if ring_ext:
        lms.extend([(13, mcp_x, mcp_y), (14, mcp_x + 5, 245), (15, mcp_x + 10, 195), (16, mcp_x + 15, 145)])
    else:
        lms.extend([(13, mcp_x, mcp_y), (14, mcp_x, 325), (15, mcp_x, 345), (16, mcp_x, 365)])

    # Pinky: 17: MCP, 18: PIP, 19: DIP, 20: TIP
    mcp_x, mcp_y = 360, 320
    if pinky_ext:
        lms.extend([(17, mcp_x, mcp_y), (18, mcp_x + 10, 260), (19, mcp_x + 15, 215), (20, mcp_x + 20, 170)])
    else:
        lms.extend([(17, mcp_x, mcp_y), (18, mcp_x, 335), (19, mcp_x, 350), (20, mcp_x, 365)])

    # If pinch is specified, place index tip close to thumb tip
    if pinch:
        # Move index tip (8) next to thumb tip (4)
        thumb_tip = lms[4]
        lms[8] = (8, thumb_tip[1] + 10, thumb_tip[2] + 10)

    return lms


class TestGestureRecognizer(unittest.TestCase):
    """Test suite for gesture classification logic."""

    def setUp(self):
        self.recognizer = GestureRecognizer(history_length=3, stability_threshold=0.6)

    def test_open_palm(self):
        """1. Verify Open Palm: All 5 fingers extended."""
        hand = make_hand(thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)
        # Feed consecutive frames to verify stability
        for _ in range(3):
            result = self.recognizer.recognize(hand, "Right")
        self.assertEqual(result.gesture, GestureType.OPEN_PALM)
        self.assertTrue(result.finger_states.thumb)
        self.assertTrue(result.finger_states.index)
        self.assertTrue(result.finger_states.middle)
        self.assertTrue(result.finger_states.ring)
        self.assertTrue(result.finger_states.pinky)

    def test_closed_fist(self):
        """2. Verify Closed Fist: All 5 fingers folded."""
        hand = make_hand(thumb_ext=False, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False)
        for _ in range(3):
            result = self.recognizer.recognize(hand, "Right")
        self.assertEqual(result.gesture, GestureType.CLOSED_FIST)
        self.assertEqual(result.finger_states.count_extended(), 0)

    def test_thumbs_up(self):
        """3. Verify Thumbs Up: Thumb pointing up, other 4 fingers curled."""
        hand = make_hand(thumb_ext=True, thumb_dir="up", index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False)
        for _ in range(3):
            result = self.recognizer.recognize(hand, "Right")
        self.assertEqual(result.gesture, GestureType.THUMBS_UP)

    def test_thumbs_down(self):
        """4. Verify Thumbs Down: Thumb pointing down, other 4 fingers curled."""
        hand = make_hand(thumb_ext=True, thumb_dir="down", index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False)
        for _ in range(3):
            result = self.recognizer.recognize(hand, "Right")
        self.assertEqual(result.gesture, GestureType.THUMBS_DOWN)

    def test_index_finger_up(self):
        """5. Verify Index Finger Up: Index extended, Middle, Ring, Pinky folded."""
        hand = make_hand(thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False)
        for _ in range(3):
            result = self.recognizer.recognize(hand, "Right")
        self.assertEqual(result.gesture, GestureType.INDEX_UP)
        self.assertTrue(result.finger_states.index)
        self.assertFalse(result.finger_states.middle)

    def test_victory_sign(self):
        """6. Verify Victory Sign: Index and Middle extended, Ring and Pinky folded."""
        hand = make_hand(thumb_ext=False, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
        for _ in range(3):
            result = self.recognizer.recognize(hand, "Right")
        self.assertEqual(result.gesture, GestureType.VICTORY)
        self.assertTrue(result.finger_states.index)
        self.assertTrue(result.finger_states.middle)
        self.assertFalse(result.finger_states.ring)
        self.assertFalse(result.finger_states.pinky)

    def test_pinch_gesture(self):
        """7. Verify Pinch Gesture: Thumb and Index tips close/touching."""
        hand = make_hand(thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True, pinch=True)
        for _ in range(3):
            result = self.recognizer.recognize(hand, "Right")
        self.assertEqual(result.gesture, GestureType.PINCH)
        self.assertLess(result.pinch_distance, 0.35)


if __name__ == "__main__":
    unittest.main()
