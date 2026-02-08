"""Generate Video Script for presentation (≤5 minutes).

Creates a concise script suitable for video presentation or pitch.
"""

import logging
from datetime import datetime
from pathlib import Path

from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def generate_video_script() -> str:
    """Generate video script markdown content.

    Returns:
        Video script content as markdown string
    """
    config = load_config()

    content = """# Video Script: Water Quality SA Predictor
## Duration: ~5 minutes

**Generated:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

---

### SCENE 1: Opening Hook (0:00-0:30)
**[Visual: Aerial footage of South African rivers and wetlands]**

**NARRATOR:**
"Water is life. But across South Africa, monitoring water quality is expensive, slow, and limited in scale. What if we could predict water contamination before it happens? What if satellites could tell us what's happening in every river, lake, and reservoir—instantly and at scale?"

**[Title Card: "Water Quality SA Predictor - AI-Powered Environmental Monitoring"]**

---

### SCENE 2: The Problem (0:30-1:15)
**[Visual: Split screen - laboratory testing vs. limited monitoring stations map]**

**NARRATOR:**
"Traditional water quality monitoring faces three major challenges:

First, **cost**. Every lab test costs 200 to 500 rand—multiplied across hundreds of locations, this becomes unsustainable.

Second, **speed**. Results take up to a week. By the time contamination is detected, the damage may already be done.

Third, **coverage**. With only about 1,000 monitoring stations nationwide, vast areas remain unmonitored."

**[Visual: Data visualization showing gaps in coverage]**

---

### SCENE 3: The Solution (1:15-2:30)
**[Visual: Satellite imagery transitioning to ML predictions overlay]**

**NARRATOR:**
"Introducing the Water Quality SA Predictor—an AI solution that combines satellite data with machine learning to predict three critical water parameters: Alkalinity, Electrical Conductance, and Dissolved Reactive Phosphorus.

Here's how it works:

**Step 1:** We collect Landsat satellite imagery—measuring vegetation indices like NDVI, water indices like NDWI, and built-up area indicators.

**Step 2:** We integrate TerraClimate data—monthly precipitation, runoff, vapor pressure, and water deficit patterns.

**Step 3:** Our machine learning ensemble—powered by XGBoost and Ridge blending—predicts water quality with spatial generalization. This means predictions work even for locations we've never monitored before."

**[Visual: Pipeline diagram showing data flow]**

---

### SCENE 4: Technical Innovation (2:30-3:15)
**[Visual: Side-by-side comparison of prediction vs. actual measurements]**

**NARRATOR:**
"What makes this solution unique?

**Anti-leakage design:** We only use satellite scenes from before or at the sample date—never future data. This ensures real-world deployability.

**Spatial validation:** We validate using leave-location-out cross-validation. If our model hasn't seen a monitoring station during training, can it still predict accurately? Yes—with an average R-squared of over 0.70 across all targets.

**100% open data:** Every data source is freely available. No licensing costs, ever."

**[Visual: Performance metrics dashboard]**

---

### SCENE 5: Business Impact (3:15-4:00)
**[Visual: Customer use cases - water utilities, agriculture, conservation]**

**NARRATOR:**
"Who benefits?

**Environmental agencies** can monitor thousands of locations monthly—without deploying a single field team.

**Water utilities** detect potential contamination early, preventing costly cleanup.

**Agricultural operations** optimize irrigation and fertilizer use based on real-time water quality insights.

**Conservation groups** track ecosystem health across protected wetlands.

Our SaaS model starts at 5,000 rand per month—delivering predictions at a fraction of traditional lab costs."

**[Visual: ROI comparison chart]**

---

### SCENE 6: The Vision (4:00-4:40)
**[Visual: Expanding map coverage across Southern Africa]**

**NARRATOR:**
"This is just the beginning.

Phase 1: We're piloting with water utilities across South Africa.

Phase 2: We'll expand coverage to Botswana, Namibia, and Zimbabwe.

Phase 3: We'll add predictive alerts—forecasting contamination events days before they occur.

Our mission? Make high-quality water monitoring accessible to every community, everywhere."

**[Visual: Children drinking clean water, flourishing ecosystems]**

---

### SCENE 7: Call to Action (4:40-5:00)
**[Visual: Logo and contact information]**

**NARRATOR:**
"Join us in protecting one of our most precious resources.

Partner with us for a pilot program.  
Integrate our API into your monitoring system.  
Or simply follow our journey on GitHub.

Together, we can ensure clean water for future generations.

**Water Quality SA Predictor—Predicting today, protecting tomorrow."**

**[End screen: Website, email, GitHub link]**

---

## Technical Notes for Video Production

### Visuals Needed
- Drone footage of South African water bodies
- Landsat/satellite imagery B-roll
- Screen recordings of prediction dashboard
- Animated pipeline diagrams
- Performance metric visualizations
- Customer testimonial clips (if available)

### Tone & Style
- Professional yet accessible
- Emphasize both technical rigor and business value
- Use data visualizations to support claims
- Keep technical jargon minimal (explain when used)

### Music Suggestions
- Opening: Uplifting, hopeful instrumental
- Problem section: Subtle tension
- Solution section: Building momentum
- Closing: Inspirational, forward-looking

### Timing Breakdown
| Section          | Duration | Cumulative |
|------------------|----------|------------|
| Hook             | 0:30     | 0:30       |
| Problem          | 0:45     | 1:15       |
| Solution         | 1:15     | 2:30       |
| Innovation       | 0:45     | 3:15       |
| Business Impact  | 0:45     | 4:00       |
| Vision           | 0:40     | 4:40       |
| Call to Action   | 0:20     | 5:00       |

---

**Script Version:** 1.0  
**Contact:** team@ey-challenge.example.com
"""

    return content


def main():
    """Generate and save Video Script."""
    setup_logging()
    logger.info("Generating Video Script...")

    config = load_config()
    content = generate_video_script()

    # Save to artifacts
    artifacts_dir = Path(config.get("paths", {}).get("artifacts", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_path = artifacts_dir / "VIDEO_SCRIPT.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Video Script saved: {output_path}")
    print(content)


if __name__ == "__main__":
    main()
