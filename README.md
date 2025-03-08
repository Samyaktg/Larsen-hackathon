# AI-Powered Content Moderation App

## TECHgium® 8th Edition Submission

### Team Members
- **Khush Trivedi**
- **Samyak Bhansali** 
- **Dhruv Kapur**
- **Mentor:** Siba Panda

## Challenge Statement
**Problem Statement:**  
Develop a real-time content moderation system that analyzes video streams to detect and address unsuitable or restricted content. The system should ensure automated precision while allowing human oversight, complying with international copyright regulations and content standards.

## Solution Overview
Our AI-powered application is designed to:
- Run multiple AI models in real-time for video and image analysis.
- Automatically detect and flag non-compliant content with precise timestamps.
- Identify a wide range of restricted materials, including:
  - **Copyrighted videos**
  - **Logos**
  - **NSFW content**
  - **Copyrighted audio**
- Utilize machine learning and deep learning for accurate detection.
- Support both **local** and **cloud-based** deployment for scalability.

## Demo Video
[![Demo Video](https://img.youtube.com/vi/otvJiEupU9w/maxresdefault.jpg)](https://www.youtube.com/watch?v=otvJiEupU9w)

## System Architecture
### **Multi-Model Content Moderation Pipeline:**
1. **Video Input**: Process videos from various sources.
2. **Keyframe Extraction**: PCA-based keyframe selection to reduce redundancy.
3. **Image Analysis**:
   - **YOLO** for object detection (e.g., identifying logos).
   - **Selenium + ResNet** for further classification.
4. **Audio Analysis**:
   - **Whisper** for speech-to-text transcription.
   - **Wav2Vec2** for inappropriate language filtering.
5. **Content Flagging**:
   - Flagged timestamps generated for non-compliant content.
   - Results combined for review and moderation.

## Key Features & Advantages
- **Real-time, multi-model processing** for efficient content moderation.
- **Scalable** architecture to support large-scale platforms.
- **Precise timestamping** for quick review and moderation.
- **Optimized for local and cloud deployment** to ensure flexibility.

## Implementation & Feedback Integration
- Improved **resource optimization** to enable local device execution.
- Modular and scalable **backend architecture**.
- Designed for **easy adaptability** to evolving requirements.

## SWOT Analysis
| Strengths | Weaknesses |
|-----------|------------|
| Real-time, multi-model processing | High computational costs |
| Automated and scalable for large platforms | Integration complexity |
| Supports various input sources | Potential for false positives/negatives |
| Provides precise timestamped flagging | Requires frequent updates for compliance |

| Opportunities | Threats |
|--------------|---------|
| Growing demand for AI-driven content moderation | Competition from tech giants |
| Partnerships with social media/streaming platforms | Legal & ethical concerns |
| Advancements in AI to improve accuracy | Constant need for updates to counter evasion |

## Roadmap: From PoC to MVP
1. **Refine core functionalities** for detecting copyrighted media, NSFW content, logos, and audio violations.
2. **Optimize model performance** to reduce computational costs and enhance accuracy.
3. **Develop a lightweight UI & API** for content review and easy integration.
4. **Deploy on a scalable cloud infrastructure** (AWS/GCP/Azure).
5. **Pilot testing** to validate accuracy and performance.
6. **Monetization strategy** through subscriptions and enterprise partnerships.
7. **Future enhancements** to improve compliance coverage and automation.

## Conclusion
Our project provides an efficient AI-powered content moderation system capable of real-time analysis for large-scale platforms. With a scalable and adaptable architecture, the system ensures compliance with international regulations while maintaining content integrity. Through strategic optimizations and phased feature rollouts, the project is well-positioned for real-world deployment, monetization, and future enhancements.

## References
- [IIITD Research on Content Moderation](https://repository.iiitd.edu.in/jspui/bitstream/handle/123456789/80/M.Tech%20Thesis%20Report-%20Swati%20Agrawal%20MT%2011034.pdf?sequence=1&isAllowed=y)
- [LSPD Dataset for Pornographic Content Detection](https://www.researchgate.net/publication/358734386_LSPD_A_Large-Scale_Pornographic_Dataset_for_Detection_and_Classification)
