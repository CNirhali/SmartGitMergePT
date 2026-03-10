## 2024-05-22 - [Accessibility & Color Contrast]
**Learning:** Using low-contrast colors like `#aaa` for status text on light backgrounds can fail accessibility standards.
**Action:** Use higher contrast colors (e.g., `#767676` or darker) for secondary text to ensure readability for all users.

## 2025-05-15 - [Dashboard Accessibility and Feedback]
**Learning:** WCAG AA compliance for text contrast (ratio 4.5:1) typically requires a gray shade like #767676 on white backgrounds, and simple CSS hover states significantly improve interactive scannability.
**Action:** Always check color contrast for "muted" or "absent" text states and add hover effects for table-based data.

## 2026-03-04 - [Domain-Specific Semantic Coloring]
**Learning:** In conflict-monitoring dashboards, "Present" (detected conflicts) should be semantically red and "Absent" (clear) should be green, which may flip standard "active/inactive" conventions.
**Action:** Always map CSS semantic classes to the user's emotional/actionable context (e.g., Warning/Danger for presence of problems) rather than just technical existence.

## 2026-03-06 - [Cross-Highlighting and Keyboard Shortcuts]
**Learning:** In data-heavy dashboards, "cross-highlighting" (visual correlation between different sections) significantly reduces cognitive load. Providing standard keyboard shortcuts like 'Escape' to clear search/filter inputs aligns with user expectations for professional tools.
**Action:** Use data attributes to link related elements in the DOM for efficient JavaScript-based highlighting, and always implement 'Escape' to clear/blur active search inputs.

## 2026-03-10 - [Visual Conflict Cues & Accessible Status]
**Learning:** In branch-heavy monitoring dashboards, providing a "secondary" visual cue like a status dot (`•`) next to branch tags in the primary list helps users identify points of interest without scanning the entire data table. Always complement these visual cues with updated ARIA labels (e.g., " (has conflicts)") to ensure accessibility parity.
**Action:** Implement subtle pseudo-element indicators for status-driven tags and always append parenthetical status text to the `aria-label` of those same tags.
