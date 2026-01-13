# Copilot Instructions

## Code Generation Philosophy

Generate minimal, synthetic code. Write only what is necessary.

**Emphasis on synthesis and deletion**: Actively remove unnecessary code, files, and documentation. Simplification through elimination is as important as adding features.

## Code Style

- Prioritize explicit, concise naming over comments
- Use comments sparingly, only when naming cannot convey intent
- No emojis
- Split logic into focused functions
- Keep functions small and single-purpose

## Development Approach

- User steers development actively
- Generate small, pertinent code increments
- Be prudent: avoid over-anticipation
- **Never create markdown files** unless explicitly requested by user
- **Do not create progress reports, summaries, or analysis documents**
- Ask user for clarification when needed
- Minimize throwaway code and rollbacks
- **Actively delete redundant or obsolete code**

## Interaction Pattern

- Create TODO.md files to communicate with user **only when explicitly needed**
- Use TODO files to:
  - Request missing data or specifications
  - Present concise options for user decision
  - Suggest next steps without implementing them
- Keep suggestions brief and actionable
- **Prefer direct communication over document creation**

## Code Organization

- One function per logical operation
- Separate concerns into distinct modules
- Avoid premature abstraction
- Implement only requested features
- **Isolate working functionality in dedicated directories**
- **Strong emphasis on decoupling and simplification**

## Quality Over Quantity

- Small, correct code beats large, approximate code
- Verify assumptions before expanding scope
- Wait for user guidance on ambiguous requirements
- **Always consider: can this be deleted or simplified?**
- **Synthesis through subtraction, not just addition**
