## 2024-05-22 - [Accessibility & Color Contrast]
**Learning:** Using low-contrast colors like `#aaa` for status text on light backgrounds can fail accessibility standards.
**Action:** Use higher contrast colors (e.g., `#767676` or darker) for secondary text to ensure readability for all users.

## 2025-05-15 - [Dashboard Accessibility and Feedback]
**Learning:** WCAG AA compliance for text contrast (ratio 4.5:1) typically requires a gray shade like #767676 on white backgrounds, and simple CSS hover states significantly improve interactive scannability.
**Action:** Always check color contrast for "muted" or "absent" text states and add hover effects for table-based data.

## 2026-03-04 - [Domain-Specific Semantic Coloring]
**Learning:** In conflict-monitoring dashboards, "Present" (detected conflicts) should be semantically red and "Absent" (clear) should be green, which may flip standard "active/inactive" conventions.
**Action:** Always map CSS semantic classes to the user's emotional/actionable context (e.g., Warning/Danger for presence of problems) rather than just technical existence.
