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

## 2026-03-12 - [Category Filtering via Hidden Text]
**Learning:** For dashboards with a global text-based filter, categories or "scenario types" can be made interactive by injecting hidden descriptive text (using an `.sr-only` utility class) into the relevant data rows. This allows a simple text search mechanism to function as a powerful categorical filter without complex conditional logic in the search implementation.
**Action:** Use hidden labels like `<span class="sr-only">Category Name</span>` inside table cells to enable instant categorical filtering when the search input is populated with that name.

## 2026-03-15 - [Bidirectional Data Highlighting]
**Learning:** Linking summary metrics to specific data rows via bidirectional highlighting (Scenario Types <-> Table Rows) provides immediate visual validation of the summary data and helps users quickly navigate large datasets. This pattern turns a static summary into an interactive "map" of the data.
**Action:** Implement `data-scenario` and `data-scenarios` attributes to link category items to data rows, and use JavaScript to toggle highlighting on both when either is hovered or focused.

## 2026-03-20 - [Universal Interactive Filtering]
**Learning:** Transforming static status badges in a data table into interactive filter triggers creates a powerful "drill-down" experience. Users intuitively expect that clicking a highlighted status (like a "Conflict" badge) will filter the view to similar items.
**Action:** Implement a centralized `applyGlobalFilter` function to handle toggling, event dispatching, and scroll management for all interactive filterable elements (branch tags, scenario types, and status badges).

## 2026-03-21 - [Empty State Recovery Path]
**Learning:** For data-heavy dashboards, "No Results" states should not just show static text; they must provide an immediate recovery path (like a "Clear filter" button) to prevent user frustration. Additionally, empty state detection must account for all relevant filtered lists (e.g., both a primary table and a secondary list) to avoid showing "No results" prematurely when some items are still visible.
**Action:** Always include a contextual "Clear/Reset" button in empty states and implement comprehensive visibility checks that encompass all filtered UI regions before rendering the "No results" feedback.

## 2026-03-22 - [Actionable Row-Level Commands]
**Learning:** In data-heavy tables, exposing common terminal-based actions (like `git diff`) as one-click "Copy" buttons that appear only on row hover/focus significantly improves developer productivity without cluttering the primary UI. Using `:focus-within` ensures these actions remain discoverable and usable for keyboard/screen reader users.
**Action:** Implement hover-triggered "action strips" or buttons for row-specific tasks and ensure accessibility via `:focus-within` and clear ARIA labels.

## 2026-03-25 - [Risk-Based Data Prioritization]
**Learning:** In monitoring dashboards, the default sort order of data lists should reflect the user's priority of risk. Placing the "base" branch first and then sorting others by their conflict count ensures that the most critical or high-friction areas are immediately visible without filtering.
**Action:** Implement multi-level sorting that prioritizes fixed "base" items and then orders by risk/conflict metrics descending.

## 2026-03-25 - [Data Freshness Feedback]
**Learning:** Displaying only absolute timestamps (e.g., "12:03:58 UTC") forces users to perform mental math to determine data freshness. Adding a live, auto-updating relative timestamp (e.g., "(just now)") provides immediate cognitive relief and confirms the dashboard is actively monitoring the state.
**Action:** Use a JavaScript interval to update relative timestamps from an ISO datetime attribute, ensuring they stay accurate without a page refresh.

## 2026-03-25 - [Defensive Copy-Paste Quoting]
**Learning:** When providing terminal commands for users to copy, assuming simple branch names (no spaces or special characters) leads to fragile workflows. Using shell-quoting (like `shlex.quote`) as a default for all variable components in copyable commands ensures they remain functional and safe for all users, regardless of their naming conventions.
**Action:** Implement a `shquote` template filter for all generated shell commands to wrap variable components in single quotes.

## 2026-03-28 - [Context-Aware Intelligent Filtering]
**Learning:** In dashboards where a primary list acts as a legend for a secondary data table, strict text filtering can be disorienting if it hides related items. "Intelligent filtering" that keeps related entities (e.g., conflict partners) visible even if they don't match the query directly maintains situational awareness and reduces "context switching" cognitive load.
**Action:** Implement multi-pass filtering logic that identifies and preserves "partner" entities linked to currently visible primary results.

## 2026-03-28 - [Robust Selection on Focus]
**Learning:** Browser-native focus behavior can often reset the cursor position *after* a standard `focus` event listener executes, negating a simple `.select()` call. Using `setTimeout(..., 0)` or `requestAnimationFrame` ensures the selection logic runs after the browser's default positioning, providing a much more reliable "select-on-focus" experience.
**Action:** Always wrap `.select()` calls in a `setTimeout(() => ..., 0)` when triggered by a `focus` or `click` event to ensure reliability across all browsers.

## 2026-04-05 - [Summary-Driven Interactive Filtering]
**Learning:** Summary statistics (e.g., "Conflict Pairs: 1") are high-intent elements that users naturally want to interact with. Converting these into buttons that pre-fill a global filter provides an intuitive "drill-down" mechanism that reduces cognitive load and improves navigation speed without adding new UI components.
**Action:** When implementing summary badges or cards, always add `role="button"` and link them to the primary filtering or search logic to satisfy user curiosity and intent.

## 2026-03-30 - [Consistency and CSP-Compliant Dynamic Content]
**Learning:** UX consistency across similar UI elements (like branch vs. file tags) reduces user surprise. When creating dynamic UI elements in a dashboard with a strict Content Security Policy (CSP), using `innerHTML` with strings containing tags (like `<kbd>`) can trigger "Unexpected identifier" errors or policy violations.
**Action:** Always prioritize functional consistency for interactive tags. Use `document.createElement` and `textContent` instead of `innerHTML` for CSP-compliant dynamic content generation.

## 2026-04-06 - [Card-Based Scenario Prioritization]
**Learning:** In dashboards where multiple "risk scenarios" are monitored, presenting them as a scannable grid of interactive cards (with explicit status badges) is significantly more effective than a simple list. This pattern provides immediate visual feedback on which categories require attention and creates a clear, high-intent interaction target for filtering.
**Action:** Use card-based layouts for category/scenario overviews and include clear, color-coded status badges (e.g., Warning/Clear) to drive user attention and interaction.
