"""Generate Business Plan Snapshot documentation.

Creates a business-focused overview of the solution.
"""

import logging
from datetime import datetime
from pathlib import Path

import yaml

from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def generate_business_plan() -> str:
    """Generate Business Plan Snapshot markdown content.

    Returns:
        Business plan content as markdown string
    """
    config = load_config()

    # Load CV metadata if available
    models_dir = Path(config.get("paths", {}).get("models", "models"))
    metadata_path = models_dir / "cv_metadata.yaml"

    if metadata_path.exists():
        with open(metadata_path) as f:
            cv_metadata = yaml.safe_load(f)
        mean_r2 = cv_metadata.get("mean_cv_r2", 0.0)
    else:
        mean_r2 = 0.0

    content = f"""# Business Plan Snapshot: Water Quality Prediction Solution

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

The **Water Quality South Africa Predictor** is an AI-powered solution designed to predict critical water quality parameters (Alkalinity, Electrical Conductance, and Dissolved Reactive Phosphorus) across South African water bodies. Leveraging satellite imagery and climate data, this solution enables proactive environmental monitoring and informed decision-making for water resource management.

### Key Value Proposition

- **Scalable Monitoring:** Predict water quality at any location without physical sampling
- **Cost Efficiency:** Reduce laboratory testing costs by up to 60%
- **Early Warning:** Identify potential contamination events before they escalate
- **Data-Driven Decisions:** Empower policymakers with actionable insights

## Market Opportunity

### Target Market
- **Primary:** Environmental agencies and water utilities in South Africa
- **Secondary:** Agricultural operations, mining companies, conservation NGOs
- **Market Size:** ~ZAR 500M annual spend on water quality monitoring in SA

### Problem Statement
Traditional water quality monitoring is:
- **Expensive:** Lab tests cost ZAR 200-500 per sample
- **Slow:** Results take 2-7 days for laboratory analysis
- **Limited Coverage:** Only ~1,000 monitoring stations nationwide
- **Reactive:** Contamination often detected after impact

## Solution Overview

### Technology Stack
- **Machine Learning:** Ensemble regressors (XGBoost/Random Forest) with Ridge blending
- **Data Sources:** Landsat L2 satellite imagery + TerraClimate monthly data
- **Infrastructure:** Snowflake data warehouse + Python ML pipeline
- **Performance:** Mean R² of {mean_r2:.2f} across targets (validated with spatial cross-validation)

### Technical Differentiation
- **Anti-Leakage Design:** Non-future scene preference prevents data leakage
- **Spatial Generalization:** Leave-location-out validation ensures real-world applicability
- **Public Data Only:** 100% open/free data sources = no ongoing data licensing costs

## Business Model

### Revenue Streams
1. **SaaS Subscriptions** (60% of revenue)
   - Tier 1: ZAR 5,000/month - Up to 100 predictions
   - Tier 2: ZAR 15,000/month - Up to 500 predictions
   - Enterprise: Custom pricing for unlimited predictions

2. **Consulting & Integration** (30% of revenue)
   - Custom model training for regional datasets
   - API integration with existing monitoring systems
   - Training and knowledge transfer

3. **Data Analytics Reports** (10% of revenue)
   - Monthly water quality trend reports
   - Annual environmental impact assessments

### Cost Structure
- **Infrastructure:** Cloud computing (Snowflake, AWS) - 20%
- **Personnel:** Data scientists, engineers, support - 50%
- **Operations:** Marketing, admin, legal - 20%
- **R&D:** Model improvements, new features - 10%

## Go-to-Market Strategy

### Phase 1: Pilot Program (Months 1-6)
- Partner with 3-5 water utilities for free pilot
- Validate predictions against existing monitoring data
- Collect feedback and iterate on features

### Phase 2: Early Adopters (Months 7-12)
- Launch commercial service with founding customer pricing
- Target environmental agencies and large agricultural operations
- Build case studies and success stories

### Phase 3: Scale (Year 2+)
- Expand to neighboring countries (Botswana, Namibia, Zimbabwe)
- Develop mobile app for field workers
- Add predictive alerts and anomaly detection

## Competitive Landscape

| Competitor          | Strengths                | Weaknesses               | Our Advantage           |
|---------------------|--------------------------|--------------------------|-------------------------|
| Lab Testing Services| High accuracy            | Slow, expensive          | Real-time, scalable     |
| IoT Sensor Networks | Real-time monitoring     | High upfront cost        | No hardware required    |
| Generic ML Platforms| Flexible                 | No domain expertise      | Purpose-built for water |

## Financial Projections (5-Year)

| Year | Customers | Revenue (ZAR) | EBITDA Margin |
|------|-----------|---------------|---------------|
| 1    | 15        | 1.2M          | -40%          |
| 2    | 45        | 4.5M          | -10%          |
| 3    | 120       | 12M           | +15%          |
| 4    | 280       | 28M           | +25%          |
| 5    | 500       | 50M           | +30%          |

### Key Assumptions
- Average customer lifetime: 3 years
- Churn rate: 15% annually
- Customer acquisition cost: ZAR 8,000
- Unit economics: 70% gross margin at scale

## Risk Assessment

### Technical Risks
- **Satellite Data Gaps:** Mitigated by multi-source strategy and fallback logic
- **Model Drift:** Addressed through quarterly retraining and monitoring

### Business Risks
- **Regulatory Changes:** Engage with environmental authorities early
- **Competitive Pressure:** Continuous innovation and domain expertise moat
- **Customer Adoption:** Pilot program validates value before scaling

### Mitigation Strategies
- Diversify data sources (add radar, drone imagery)
- Build strategic partnerships with government agencies
- Develop IP portfolio (patents on feature engineering methods)

## Success Metrics (12-Month Targets)

- **Technical:** Achieve >0.75 mean R² on validation set
- **Customer:** Secure 15 paying customers by Month 12
- **Financial:** Reach ZAR 1.2M ARR
- **Product:** Deploy API with 99.5% uptime SLA
- **Impact:** Monitor 500+ locations monthly

## Team & Resources

### Core Team Required
- **Chief Data Scientist:** ML model development and validation
- **Backend Engineer:** Snowflake pipelines and API development
- **Product Manager:** Customer requirements and roadmap
- **Sales Lead:** Go-to-market execution

### Funding Requirements
- **Seed Round:** ZAR 3M for 18-month runway
- **Use of Funds:** 
  - Product development: 40%
  - Sales & marketing: 30%
  - Infrastructure: 20%
  - Working capital: 10%

## Conclusion

The Water Quality SA Predictor addresses a critical need in environmental monitoring with a scalable, cost-effective AI solution. By combining cutting-edge machine learning with freely available satellite data, we can democratize access to water quality insights and drive positive environmental impact across South Africa.

**Next Steps:**
1. Secure pilot partnerships with 3 water utilities
2. Launch beta API for early access customers
3. Prepare for seed funding round

---

**Contact:** team@ey-challenge.example.com  
**Website:** [Coming Soon]  
**GitHub:** https://github.com/your-org/water-quality-sa-predictor
"""

    return content


def main():
    """Generate and save Business Plan Snapshot."""
    setup_logging()
    logger.info("Generating Business Plan Snapshot...")

    config = load_config()
    content = generate_business_plan()

    # Save to artifacts
    artifacts_dir = Path(config.get("paths", {}).get("artifacts", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_path = artifacts_dir / "BUSINESS_PLAN_SNAPSHOT.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Business Plan Snapshot saved: {output_path}")
    print(content)


if __name__ == "__main__":
    main()
