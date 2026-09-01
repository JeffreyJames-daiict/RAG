"""Unit tests for spatial caption detection and association."""

import pytest
from rag.assets.caption_detector import CaptionDetector


def test_caption_regex_matching():
    detector = CaptionDetector()
    assert detector.is_caption_text("Figure 1: Overall System Architecture") is True
    assert detector.is_caption_text("Fig. 2 - Proposed Neural Network") is True
    assert detector.is_caption_text("Diagram 3. Data Processing Pipeline") is True
    assert detector.is_caption_text("Chart 4: Latency vs Throughput") is True
    assert detector.is_caption_text("This is regular paragraph text explaining the figure.") is False


def test_spatial_caption_association():
    detector = CaptionDetector(vertical_tolerance_pts=70.0)

    figure_bbox = (50.0, 100.0, 350.0, 300.0)  # y: 100 to 300

    captions = [
        ((50.0, 315.0, 350.0, 330.0), "Figure 1: System Design"),  # 15 pts below figure
        ((50.0, 700.0, 350.0, 720.0), "Figure 2: Distant Chart"),  # 400 pts below
    ]

    associated = detector.associate_caption_to_figure(figure_bbox, captions)
    assert associated == "Figure 1: System Design"


def test_spatial_caption_association_above_figure():
    detector = CaptionDetector(vertical_tolerance_pts=70.0)

    figure_bbox = (50.0, 200.0, 350.0, 400.0)  # y: 200 to 400

    captions = [
        ((50.0, 175.0, 350.0, 195.0), "Figure 3: Architecture Diagram (Top Caption)"),  # 5 pts above figure
    ]

    associated = detector.associate_caption_to_figure(figure_bbox, captions)
    assert associated == "Figure 3: Architecture Diagram (Top Caption)"
