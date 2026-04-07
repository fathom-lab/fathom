# FATHOM for Education

## The EdTech AI Problem

The global AI in education market is projected to reach $30B+ by 2027. AI tutoring is everywhere:
- Khan Academy deployed Khanmigo (GPT-4 tutor)
- Duolingo uses AI for adaptive lessons
- Chegg launched CheggMate AI
- Coursera, edX, and Udemy all have AI features

The problem nobody has solved: **how do you know if the AI tutor is actually teaching reasoning, or just giving answers?**

When a student asks "why does photosynthesis work?", the AI should reason through the mechanism. If it pattern-matches a textbook answer, the student gets information but not understanding. Parents, teachers, and institutions have no way to tell the difference.

## Fathom for Education

### The Parent/Teacher Dashboard Concept

A dashboard that shows:
- **Reasoning depth** of AI tutor responses to each student
- **Trend over time** — is the AI engaging deeper reasoning as questions get harder?
- **Subject-level breakdown** — which subjects show genuine AI reasoning vs pattern completion?
- **Alert system** — flag when AI responses drop below a reasoning threshold

### Use Cases

**1. AI Tutor Certification**
Before an AI tutor is deployed in a school:
```
Benchmark: 50 standardized questions across subjects
Fathom scores each AI response
Report: "This AI tutor demonstrates B+ average reasoning depth.
72% of responses show distributed analytical processing.
Recommended for: supplementary tutoring with teacher oversight."
```

**2. Assignment Integrity**
When a student submits AI-assisted work, Fathom scores the AI component. Teachers see whether the AI was used for genuine reasoning assistance or just answer generation.

**3. Curriculum Quality**
Score AI-generated course materials. Materials that show deep reasoning depth teach better mental models. Materials that show shallow depth are memorization-focused.

**4. Adaptive Depth**
The AI tutor adjusts based on Fathom self-scoring. If the student asks a question that produces a shallow AI response, the tutor automatically elaborates with deeper reasoning.

### Sample API Call
```bash
curl -X POST https://api.fathom.darkflobi.com/v2/compare \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "surface_prompt": "Photosynthesis is the process by which plants convert sunlight into",
    "insight_prompt": "Photosynthesis converts light energy into chemical energy because chlorophyll absorbs specific wavelengths and the excited electrons drive a chain of redox reactions that ultimately"
  }'
```

## Pricing: Education Tier

| Tier | Price | Includes |
|------|-------|---------|
| Teacher / Individual | $9/month | 1k scores, basic dashboard |
| School / District | $99/month | 20k scores, teacher dashboard, student reports |
| University | $499/month | 100k scores, LMS integration, research API access |
| EdTech Platform | $1,499/month | 500k scores, white-label, batch API, custom integration |

## Partnership Angles

- **Canvas / Blackboard / Moodle:** Plugin that scores AI-generated content within the LMS
- **Khan Academy / Khanmigo:** Depth scoring integrated into tutor responses
- **College Board / ACT:** AI content verification for test prep materials
- **Turnitin:** Complement plagiarism detection with reasoning depth analysis

## Target Customers

- K-12 school districts with AI tutor programs
- Universities evaluating AI in curriculum
- EdTech companies (Khan Academy, Coursera, Duolingo, Chegg)
- Education ministries setting AI policy
- Tutoring companies (Varsity Tutors, Wyzant) adding AI
