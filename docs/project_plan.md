# AI-Powered Dermatology Boards Prep Platform

This document outlines the high-level plan for building the AI-Powered Dermatology Boards Prep Platform. It is intended as a living overview and will evolve with the project.

## 1. Goals
- Provide dermatology residents, fellows, and medical students with an adaptive question bank.
- Leverage a multi-agent AI system to generate high-quality exam-style questions.
- Offer robust admin tools for reviewing, curating, and managing content.
- Ensure security and scalability using Firebase services and Google AI Platform.

## 2. Core Technologies
- **Frontend:** Next.js (React)
- **Backend:** Cloud Functions for Firebase (Node.js/Python)
- **Database:** Cloud Firestore + Vector Database for RAG
- **Authentication:** Firebase Auth
- **AI Engine:** Google Gemini models
- **Hosting:** Firebase Hosting

## 3. Major Milestones
1. **Repository Setup**
   - Configure project structure and CI/CD (if applicable).
   - Establish linting, formatting, and testing frameworks.

2. **Authentication & User Management**
   - Implement email/password and Google sign-in via Firebase Auth.
   - Create Firestore `users` collection with the schema described in the project spec.
   - Role-based access control for `learner` vs `admin`.

3. **Database Schema**
   - Define collections for `topics`, `questions`, `cases`, `feedback`, etc.
   - Set up Firestore security rules to enforce per-user access.

4. **AI Question Generation Pipeline**
   - Implement Cloud Functions orchestrating the multi-agent generation workflow.
   - Integrate Gemini models with defined prompts and validation via Zod schemas.
   - Store generated questions and track status (`generated`, `needs_refinement`, `reviewer_approved`).

5. **Quiz Creation & Submission**
   - Build Next.js pages for quiz setup and question presentation.
   - Implement API endpoints for quiz generation and quiz submission.
   - Persist quiz sessions and user answers as outlined in the architecture notes.

6. **Dashboard & Analytics**
   - Provide learners with performance stats and recent quiz history.
   - Develop admin dashboard for reviewing questions and feedback.

7. **APPLIED Exam Simulation & Advanced Features**
   - Support case-based question sequences.
   - Add adaptive mode, follow‑up questions, and review chatbot (post-MVP).

8. **Knowledge Base Management**
   - Populate `knowledge_core` collection with validated medical text.
   - Automate research updates via scheduled Cloud Functions.

9. **Testing & Quality Assurance**
   - Unit and integration tests for backend functions.
   - End-to-end testing of quiz flow and admin tools.

10. **Deployment & Monitoring**
    - Deploy to Firebase Hosting and configure Cloud Functions triggers.
    - Set up logging, error reporting, and analytics.

## 4. Ongoing Tasks
- Maintain changelogs in the `logs/` directory for every update.
- Capture evolving user preferences in `preferences.md`.
- Review and refine this plan as requirements change.

