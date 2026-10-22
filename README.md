# SerpentGuard AI: snake detection and classification

Group 5, DS and AI Lab Project (IIT Madras BS, DS & AI Lab course).

SerpentGuard AI finds a snake in a photo or a live camera feed and suggests its species from a fixed list of 27 Indian species. It shows the WHO medical-importance category of each candidate and says "uncertain" or "unknown" when it is not confident. Live video runs the same models as photos, one frame at a time: a YOLOv8n detector, ByteTrack tracking, a MobileViT classifier, and smoothing over recent frames. No LLM or internet API is involved.

> Safety: this is a student research and education project. It does not diagnose bites or give treatment advice, and "no snake detected" never means an area is safe. In an emergency, go to the nearest hospital.

## Team

| Member | Role |
|---|---|
| Jayesh Bansal | Team lead |
| Vikas | Member |
| Puneet Bajaj | Member |
| Manas Rastogi | Member |

Each member's weekly contributions are in [work-contribution-log.md](work-contribution-log.md).

## Milestones

| # | Milestone | Due (2026) | Status | Documents |
|---|---|---|---|---|
| 1 | Problem definition and literature review | 22 Oct | Submitted | [Report](documentation/milestone%201/project-report.md) · [Slides](documentation/milestone%201/assets/SerpentGuard_AI_Group5_Presentation.pdf) |
| 2 | Dataset preparation | 29 Oct | Planned | |
| 3 | Model architecture and end-to-end pipeline | 5 Nov | Planned | |
| 4 | Model training | 19 Nov | Planned | |
| 5 | Evaluation and analysis | 26 Nov | Planned | |
| 6 | Deployment and documentation | 10 Dec | Planned | |

Course requirements: [guidelines](https://bsc-iitm.github.io/dsai-lab-project/guidelines.html) · [milestones](https://bsc-iitm.github.io/dsai-lab-project/milestones.html).

## Repository layout

```text
documentation/          milestone reports and slides (one folder per milestone)
work-contribution-log.md  weekly per-member contributions
```

Training scripts, inference notebooks, configs and the data manifest arrive from Milestone 2 onward, and this README will explain how to run them.

## Data and licences

Training images come from iNaturalist research-grade observations with CC0 or CC-BY family licences, plus Open Images V7 and the other sources listed in the [Milestone 1 report](documentation/milestone%201/project-report.md#32-sources-by-task). We never commit the images. The repository holds a manifest (observation id, photo id, licence, source URL) and a script that downloads them, and each image keeps its original licence and attribution. Some of those licences are non-commercial, so the project is for academic use only.
